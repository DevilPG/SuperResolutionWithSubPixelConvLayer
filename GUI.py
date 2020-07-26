# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:38:57 2019

@author: HP-PC
"""

from tkinter import *
import os
import data_generate
import test_image
import train

class SR_GUI(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        
    def createWidgets(self):
        self.var=StringVar()
        self.var.set('3')
        self.label=Label(self,text='choose upscale factor:')
        self.label.pack()
        self.r1=Radiobutton(self,text='x2',variable=self.var,value='2',command=self.print_to_entry)
        self.r1.pack()
        self.r2=Radiobutton(self,text='x3',variable=self.var,value='3',command=self.print_to_entry)
        self.r2.pack()
        self.r3=Radiobutton(self,text='x4',variable=self.var,value='4',command=self.print_to_entry)
        self.r3.pack()
        self.r4=Radiobutton(self,text='x8',variable=self.var,value='8',command=self.print_to_entry)
        self.r4.pack()
        self.b1=Button(self,text='Generate Dateset',width=15,command=self.generate_data)
        self.b1.pack()
        self.b2=Button(self,text='Train',width=15,command=self.train)
        self.b2.pack()
        self.b3=Button(self,text='Test',width=15,command=self.test_SR)
        self.b3.pack()
        
    def print_to_entry(self):
        self.label.config(text='current upscale factor: '+self.var.get())
        
    def generate_data(self):
        print('dataset generating...')
        data_generate.main(int(self.var.get()))
        print('dataset successfully generated!')
        
        
    def train(self):
        print('model training...')
        train.main(int(self.var.get()))
        print('model trained!')
        
    def test_SR(self):
        print('testing...')
        test_image.main(int(self.var.get()))
        print('test done!')
        
        
if __name__ == "__main__":
    gui = SR_GUI()
    gui.master.title('Super Resolution')
    
    gui.mainloop()