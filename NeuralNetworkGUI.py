import pathlib
import pygubu
import pickle
import random
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
from threading import Thread

from Networks.Dense import DenseNetwork

#PROJECT_PATH = pathlib.Path(__file__).parent
#PROJECT_UI = PROJECT_PATH / "Praca_1.ui"


class NeuralNetworkGUI:
    def __init__(self, master=None):
        #MY VARS
        self.sizes = []
        self.network = None
        self.training_data = None
        self.validation_data = None
        self.prediction_data = None

        self.text_last_line = 1
        self.text_queue = queue.Queue()
        self.statistics = []


        #GENERATED
        self.window = tk.Tk()
        self.window.title('BitFortune')
        self.frame1 = ttk.Frame(self.window)
        self.labelframe1 = ttk.Labelframe(self.frame1)
        self.layer_type_cbox = ttk.Combobox(self.labelframe1)
        self.layer_type_cbox.configure(width='15')
        self.layer_type_cbox.place(anchor='nw', x='10', y='15')
        self.label1 = ttk.Label(self.labelframe1)
        self.label1.configure(text='Typ')
        self.label1.place(anchor='nw', x='10', y='-5')
        self.activation_cbox = ttk.Combobox(self.labelframe1)
        self.activation_cbox.configure(width='15')
        self.activation_cbox.place(anchor='nw', x='130', y='15')
        self.neuron_count_entry = ttk.Entry(self.labelframe1)
        self.neuron_count_entry.configure(exportselection='true')
        self.neuron_count_entry.place(anchor='nw', width='60', x='250', y='15')
        self.add_layer_btn = ttk.Button(self.labelframe1)
        self.add_layer_btn.configure(text='Pridaj')
        self.add_layer_btn.place(anchor='nw', width='65', x='320', y='12')
        self.add_layer_btn.configure(command=self.add_layer_btn_press)
        self.label2 = ttk.Label(self.labelframe1)
        self.label2.configure(text='Aktivacia')
        self.label2.place(anchor='nw', x='130', y='-5')
        self.label3 = ttk.Label(self.labelframe1)
        self.label3.configure(text='Neurony')
        self.label3.place(anchor='nw', x='250', y='-5')
        self.reset_model_btn = ttk.Button(self.labelframe1)
        self.reset_model_btn.configure(text='Reset')
        self.reset_model_btn.place(anchor='nw', width='65', x='320', y='42')
        self.reset_model_btn.configure(command=self.reset_model_btn_press)
        self.labelframe1.configure(height='115', text='Pridat vrstvu', width='400')
        self.labelframe1.place(anchor='nw', x='10', y='0')
        self.labelframe2 = ttk.Labelframe(self.frame1)
        self.load_training_btn = ttk.Button(self.labelframe2)
        self.load_training_btn.configure(text='Nacitaj trening')
        self.load_training_btn.place(anchor='nw', width='130', x='10', y='5')
        self.load_training_btn.configure(command=self.load_training_btn_press)
        self.load_predict_btn = ttk.Button(self.labelframe2)
        self.load_predict_btn.configure(text='Nacitaj predpoved')
        self.load_predict_btn.place(anchor='nw', width='130', x='10', y='40')
        self.load_predict_btn.configure(command=self.load_predict_btn_press)
        self.compile_btn = ttk.Button(self.labelframe2)
        self.compile_btn.configure(text='Compile')
        self.compile_btn.place(anchor='nw', width='100', x='150', y='5')
        self.compile_btn.configure(command=self.compile_btn_press)
        self.predict_btn = ttk.Button(self.labelframe2)
        self.predict_btn.configure(text='Predpovedaj')
        self.predict_btn.place(anchor='nw', width='100', x='150', y='40')
        self.predict_btn.configure(command=self.predict_btn_press)
        self.train_btn = ttk.Button(self.labelframe2)
        self.train_btn.configure(text='Trenuj')
        self.train_btn.place(anchor='nw', width='100', x='260', y='5')
        self.train_btn.configure(command=self.train_btn_press)
        self.epochs_entry = ttk.Entry(self.labelframe2)
        self.epochs_entry.place(anchor='nw', width='50', x='55', y='70')
        self.lr_entry = ttk.Entry(self.labelframe2)
        self.lr_entry.place(anchor='nw', width='50', x='140', y='70')
        self.epochs_label = ttk.Label(self.labelframe2)
        self.epochs_label.configure(text='Epochs:')
        self.epochs_label.place(anchor='nw', x='10', y='70')
        self.lr_label = ttk.Label(self.labelframe2)
        self.lr_label.configure(text='LR:')
        self.lr_label.place(anchor='nw', x='120', y='70')
        self.labelframe2.configure(height='115', text='Dataset a model', width='370')
        self.labelframe2.place(anchor='nw', x='420', y='0')

        self.model_treeview = ttk.Treeview(self.frame1)
        self.model_treeview_cols = ['type_col', 'activation_col', 'neuron_count_col']
        self.model_treeview_dcols = ['type_col', 'activation_col', 'neuron_count_col']
        self.model_treeview.configure(columns=self.model_treeview_cols, displaycolumns=self.model_treeview_dcols)
        
        self.model_treeview.column('#0', stretch = tk.NO, width = 0)
        self.model_treeview.column('type_col', anchor='w',stretch='false',width='133',minwidth='20')
        self.model_treeview.column('activation_col', anchor='w',stretch='true',width='133',minwidth='20')
        self.model_treeview.column('neuron_count_col', anchor='w',stretch='true',width='134',minwidth='20')
        
        self.model_treeview.heading('#0', text = '',anchor=tk.W)
        self.model_treeview.heading('type_col', anchor='w',text='Typ')
        self.model_treeview.heading('activation_col', anchor='w',text='Aktivacia')
        self.model_treeview.heading('neuron_count_col', anchor='w',text='Pocet neuronov')
        self.model_treeview.place(anchor='nw', height='300', width='400', x='10', y='125')

        self.labelframe3 = ttk.Labelframe(self.frame1)
        self.labelframe3.configure(height='200', text='Statistiky', width='200')
        self.labelframe3.place(anchor='nw', height='308', width='780', x='10', y='435')
        self.frame1.configure(height='750', width='800')
        self.frame1.pack(side='top')

        self.statistics_output = ttk.Labelframe(self.frame1)
        self.statistics_output.configure(height='200', text='Statistiky', width='200')
        self.statistics_output.place(anchor='nw', height='308', width='780', x='10', y='435')

        self.console_out = tk.Text(self.frame1)
        self.console_out.configure(height='10', width='50')
        self.console_out.place(anchor='nw', height='300', width='370', x='420', y='125')

        #self.model_treeview = ttk.Treeview(self.frame1)
        #self.model_treeview_cols = ['type_col', 'activation_col', 'neuron_count_col']
        #self.model_treeview_dcols = ['type_col', 'activation_col', 'neuron_count_col']
        #self.model_treeview.configure(columns=self.model_treeview_cols, displaycolumns=self.model_treeview_dcols)
        
        #self.model_treeview.column('#0', stretch = tk.NO, width = 0)
        #self.model_treeview.column('type_col', anchor='w',stretch='false',width='133',minwidth='20')
        #self.model_treeview.column('activation_col', anchor='w',stretch='true',width='133',minwidth='20')
        #self.model_treeview.column('neuron_count_col', anchor='w',stretch='true',width='134',minwidth='20')
        
        #self.model_treeview.heading('#0', text = '',anchor=tk.W)
        #self.model_treeview.heading('type_col', anchor='w',text='Typ')
        #self.model_treeview.heading('activation_col', anchor='w',text='Aktivacia')
        #self.model_treeview.heading('neuron_count_col', anchor='w',text='Pocet neuronov')
        #self.model_treeview.place(anchor='nw', height='300', width='400', x='10', y='75')

        self.activation_cbox['values'] = ('Sigmoid')
        self.activation_cbox.current(0)
        self.layer_type_cbox['values'] = ('Dense')
        self.layer_type_cbox.current(0)

        # Main widget
        self.mainwindow = self.frame1
    
    def run(self):
        #self.mainwindow.mainloop()
        plt.tight_layout()
        f = Figure()
        f.set_size_inches(0.5, 4)
        plot1 = f.add_subplot(111)
        plot1.set_ylabel('Chyba')
        canvas = FigureCanvasTkAgg(f, master = self.statistics_output)
        
        while True:
            if (not self.text_queue.empty()):
            #Update console text and statistics.
            #while (not self.text_queue.empty()):
                while (not self.text_queue.empty()):
                    self.console_out.insert('0.0', self.text_queue.get())

                plot1.clear()
                #plot1 = f.add_subplot(111, xlabel = 'Epoch', ylabel = 'Chyba')
                
                x = [x[0] for x in self.statistics]
                y = [x[1][0] for x in self.statistics]
            
                plot1.plot(x, y)
                plot1.set_ylabel('Chyba')

                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            self.mainwindow.update_idletasks()
            self.mainwindow.update()

    def add_layer_btn_press(self):
        n_count = self.neuron_count_entry.get()
        self.neuron_count_entry.config(text = '')
        if n_count == '':
            return
        self.sizes.append(int(n_count))
        self.model_treeview.insert('', 'end', iid = len(self.sizes), text = 'text', values = ('Dense', 'Sigmoid', n_count))

    def reset_model_btn_press(self):
        self.model_treeview.delete(*self.model_treeview.get_children())
        self.sizes = []
        self.network = None

    def load_training_btn_press(self):
        filename = askopenfilename()
        file = open(filename, 'rb')
        self.training_data = pickle.load(file)

    def load_predict_btn_press(self):
        pass

    def compile_btn_press(self):
        self.network = DenseNetwork(self.sizes)
        self.network.compile()
        print('[INFO] Siet skompilovana')
        self.text_queue.put('[INFO] Siet skompilovana\n')

    def predict_btn_press(self):
        pass

    def train_btn_press(self):
        if self.training_data is None:
            print('[ERROR]Neexistuju trenovacie data!')
            self.text_queue.put('[ERROR]Neexistuju trenovacie data!\n')
            return
        
        epochs = self.epochs_entry.get()
        lr = self.lr_entry.get()
        if epochs is '' or lr is '':
            print('[ERROR]Epochs alebo LR zadane nespravne!')
            return
        
        print('[INFO] Trening zacal.')
        self.text_queue.put('[INFO] Trening zacal.\n')
        #Train = Test for now.
        #Split GUI and training threads.
        training_thread = Thread(target = self.network.train, args = (self.training_data, self.training_data, int(epochs), float(lr), False, self.text_queue, self.statistics))
        training_thread.start()


if __name__ == '__main__':
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    app.run()


