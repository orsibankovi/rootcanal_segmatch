import math
import torch
import xlwt
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchmetrics
import metrics

class Test():
    def __init__(self, dev, batch_size, testset, net, k):
        super(Test, self).__init__()
        self.dev = dev
        self.batch_size = batch_size
        self.testset = testset
        self.net = net
        self.Dice = torchmetrics.Dice(zero_division=1.0, threshold=0.5).to(self.dev)
        self.Jaccard = torchmetrics.JaccardIndex(task='binary', threshold=0.5).to(self.dev) 
        self.criterion = nn.BCELoss().to(self.dev)
        self.k = k
        
    def test(self):
        # To open Workbook
        wb, ws = self.create_excel()

        test_loader = torch.utils.data.DataLoader(self.testset, self.batch_size, shuffle=False)
        criterion = nn.BCELoss()

        print('test' + '\n')
        test_losses, test_dice_losses, test_jaccard, fpr, tpr = [], [], [], [], []
        self.net.train(False)

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if self.dev.type == 'cuda':
                    data = data.to(self.dev)
                    target = target.to(self.dev)
                data = data.float()

                output = self.net(data)
                loss = criterion(output, target)
                
                dice_loss_value = self.Dice(output, target.int())
                jaccard_value = self.Jaccard(output, target.int())

                e = metrics.centers_of_canals(torch.round(output.cpu().data), target.cpu().int())
                test_losses.append(loss.item())
                test_dice_losses.append(dice_loss_value.item())

                if math.isnan(jaccard_value.item()):
                    test_jaccard.append(1.0)
                else:
                    test_jaccard.append(jaccard_value.item())
                    
                roc = torchmetrics.ROC(task='binary', thresholds=100)
                fpr_, tpr_, _ = roc(output, target.int())
                    
                if np.average(np.asarray(output.cpu().squeeze(1)))>0 and np.average(np.asarray(target.cpu().squeeze(1)))>0:
                    fpr.append(np.asarray(fpr_.cpu()))
                    tpr.append(np.asarray(tpr_.cpu()))

                if (i%50 == 0):
                    print('Test Loss: ' + str(round(test_losses[-1], 8)))
                    print('Test DiceLoss: ' + str(round(test_dice_losses[-1], 8)))
                    print('Test JaccardIndex: ' + str(round(test_jaccard[-1], 8)))
                    print('Euclidean distance: ' + str(e))
                    print(str(i/len(test_loader)))

                self.write_excel(ws, i, test_losses[-1], test_dice_losses[-1], test_jaccard[-1], e)
                
                
        ws.write(i+5, 1, np.average(test_losses))
        ws.write(i+5, 2, np.average(test_dice_losses))
        ws.write(i+5, 3, np.average(test_jaccard))
        
        fpr_avg = np.average(np.asarray(fpr), axis=0)
        tpr_avg = np.average(np.asarray(tpr), axis=0)
        
        wb.save('./test_results.xls')
        
        plt.figure(7)
        auc=np.trapz(tpr_avg, x=fpr_avg, dx=0.01)
        plt.plot(fpr_avg, tpr_avg, label='AUC='+str(round(auc, 4)))
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), label='baseline', linestyle='--')

        plt.title('ROC Curve', fontsize=18)
        plt.ylabel('TPR', fontsize=16)
        plt.xlabel('FPR', fontsize=16)
        plt.legend(fontsize=12, loc='lower right')
        plt.savefig('./ROC.png')
                
    def run(self):
        self.test()

    def create_excel(self) -> tuple:
        wb = xlwt.Workbook()
        ws = wb.add_sheet('sheet')
        ws.write(0, 1, 'Valid loss')
        ws.write(0, 2, 'Dice loss')
        ws.write(0, 3, 'Jaccard index')
        ws.write(0, 4, 'Euclidean dist')
        return wb, ws
    
    def write_excel(self, ws, count, loss, dice_loss, jaccard, e) -> None:
        ws.write(count+1, 1, loss)
        ws.write(count+1, 2, dice_loss)
        ws.write(count+1, 3, jaccard)
        if e != 'nan':
            ws.write(count+1, 4, e)