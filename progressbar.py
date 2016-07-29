import time
class progressbar():
    def __init__(self, iterations):
        self.iterations = iterations
        self.begin_time = 0
    def show(self, i):
        iterations = self.iterations
        begin_time = self.begin_time

        if begin_time != 0:
            eta = ((time.clock()-begin_time)/i)*iterations-i
            percent = 0.
            percent = float(i)/iterations*100
            bar = '|'+('#'*(int(percent)/2))+(' '*(50-(int(percent)/2)))+'|'
            print bar+' %d'%i+"\t"+'%.2f'%percent+'%'+'\t'+'ETA: '+'%.4f'%eta+'s'

        if begin_time == 0:
            self.begin_time = time.clock()
