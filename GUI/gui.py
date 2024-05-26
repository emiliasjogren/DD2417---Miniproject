import TrigramPredictor as tripred
from tkinter import *
from PIL import Image, ImageTk
import LoadTransformer 
import LoadRNN

class PredictorApp:
    def __init__(self):
        self.saved_keystrokes = 0
        self.total_keystrokes = 0
        self.text = []
        self.user_input = ""
        self.options = []

        #getting the trigram predictor model
        
        self.number_of_options = 1000
        root = Tk()
        self.root = root
        self.canvas_width = 800
        self.canvas_height = 600

        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.background_image = Image.open("Background.png")
        self.background_image = self.background_image.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.canvas.create_image(0, 0, anchor=NW, image=self.background_photo)
        
        self.model = None

        bigram_button = Button(self.root, text="Trigram model", command=lambda: self.set_model("Trigram"))
        bigram_button.place(x=(self.canvas_width - bigram_button.winfo_reqwidth())/2, y=320)

        rnn_button = Button(self.root, text="Recurent Neural Network",fg="#645380", command=lambda: self.set_model("RNN"))
        rnn_button.place(x=(self.canvas_width - rnn_button.winfo_reqwidth())/2, y=370)

        transformer_button = Button(self.root, text="Transformer model", command=lambda: self.set_model("Transformer"))
        transformer_button.place(x=(self.canvas_width - transformer_button.winfo_reqwidth())/2, y=420)


        

    def set_model(self, model_name):
        self.model_name = model_name
        #if model_name == "Trigram":
        self.trigram_predictor = tripred.TrigramPredictor()
        self.trigram_predictor.read_model("harry_potter_1_model.txt")
        if model_name == "Transformer":
            self.TransformerClass = LoadTransformer.LoadTransformerModel("model_Transformer.pth")
        elif model_name == "RNN":
            self.RNNClass = LoadRNN.RNN_GUI("model_RNN.pth")
        self.root.destroy()
        self.root = Tk()

        self.create_main_window()

        img = Image.open("Button.png")
        img = img.resize((600, 100)) 
        img = ImageTk.PhotoImage(img)
        background_label = Label(self.root, image=img, width=600, height=100)
        background_label.image = img  
        background_label.place(x=400, y=170, anchor=CENTER)

        self.text_label = Label(self.root, text="", font=("Helvetica", 16), bg="#645380", fg="white", wraplength=380)
        self.text_label.place(x=400, y=170, anchor=CENTER)

    def create_main_window(self):
        self.canvas_width = 800
        self.canvas_height = 600

        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.background_image = Image.open("Background.png")
        self.background_image = self.background_image.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.canvas.create_image(0, 0, anchor=NW, image=self.background_photo)

        self.word_input = StringVar()
        self.word_input.trace_add('write', self.key_event)
        self.entry = Entry(self.root, textvariable=self.word_input)
        self.entry_window = self.canvas.create_window(self.canvas_width/2, 400, window=self.entry, anchor=CENTER)
        self.entry.bind('<Return>', self.return_key_event)
        self.create_label("Enter word:", self.canvas_width/2, 370, 20)
        self.create_label("Type END to end the program", self.canvas_width/2, 425, 10)

        self.button_frame = Frame(self.root)
        self.button_frame.pack()
        

    def create_label(self, text, x, y, fontsize):
        self.canvas.create_text(x, y, text=text, font=("Helvetica", fontsize), fill="white", anchor=CENTER)


    def text_builder(self):
        return " ".join(self.text)

    def update_text_window(self): #updates the text window
        self.text_label.config(text=self.text_builder())

    def successful_guess(self, word): #if one of the predictions is the correct one
        self.saved_keystrokes += len(word) - len(self.word_input.get())
        self.total_keystrokes += len(word) - len(self.word_input.get())
        self.text.append(word)
        self.update_text_window() #add the word to our text
        #self.user_input = word #use the predicted word as input for the next prediction
        #self.text.append(self.user_input.lower())
        self.word_input.set("")
        self.button_frame.destroy()
        self.show_options(success=True)

    def unsuccessful_guess(self): #if none of the predicted words was what we wanted
        
        self.button_frame.destroy()

    def return_key_event(self, event): #what happens when the return key is pressed in the main user input window
        self.user_input = self.word_input.get()
        
        if self.user_input == "END":
            self.quit_app()
            return
        self.text.append(self.user_input.lower())
        #self.entry.delete(0, 'end') #delete the text in the entry field
        self.button_frame.destroy()
        self.word_input.set("")
        self.update_text_window() #add the word to our text
        
            
        
        self.show_options(success=False) #call the function to show the predictions

    def key_event(self,name, index, mode):
        self.user_input = self.word_input.get()
        if self.user_input == "":
            return
        self.button_frame.destroy()
        
        self.show_options_key(success=False) #call the function to show the predictions

    def quit_app(self):
        self.total_keystrokes -= 3
        print("Saved keystrokes: " + str(self.saved_keystrokes))
        print("Total keystrokes: " + str(self.total_keystrokes))
        print("Final text: \n")
        print(" ".join(self.text))
        self.root.destroy()
        #show the stats
        stats_window = Tk()
        stats_window.title("Statistics")
        stats_window.geometry("200x200")
        stats="Total keystrokes: " + str(self.total_keystrokes)+ "\n"+ "Saved keystrokes: " + str(self.saved_keystrokes) 
        text_label = Label(stats_window, text=stats, wraplength=380)
        text_label.pack(pady=20)


    def show_options(self, success=False):
        if self.model_name == "Trigram":
                self.options = self.trigram_predictor.predict_trigram(self.text, self.number_of_options)
        elif self.model_name == "Transformer":
            self.options = self.TransformerClass.get_next_k_word(self.text, self.number_of_options)
        elif self.model_name == "RNN":
            self.options = self.RNNClass.gui_generate(self.text, neighbours=self.number_of_options)

        options_fit = [word for word in self.options if word[:len(self.user_input)] == self.user_input]

        self.button_frame = Frame(self.root)
        self.button_frame.pack()

        for word in options_fit[:5]:
            Button(self.button_frame, text=word, padx=10, pady=5, font=("Helvetica", 10), command=lambda word=word: self.successful_guess(word)).pack(side='left')

        #Button(self.button_frame, text="NO CORRECT OPTION", padx=10, pady=5, font=("Helvetica", 10), command=self.unsuccessful_guess).pack(side='left')
        self.button_frame.place(x=self.canvas_width/2, y=300, anchor=CENTER)

    def show_options_key(self, success=False):
        self.total_keystrokes += 1#len(self.user_input)'

        if len(self.text) == 0: #Update options every time for first word
            self.options = self.trigram_predictor.predict_trigram("hehfhfhej", self.number_of_options)
        options_fit = [word for word in self.options if word[:len(self.user_input)] == self.user_input]
        #print(self.options)
        self.button_frame = Frame(self.root)
        self.button_frame.pack()
        for word in options_fit[:5]: #Hårdkodat nu - går att ändra
            Button(self.button_frame, text=word, padx=10, pady=5, font=("Helvetica", 10), command=lambda word=word: self.successful_guess(word)).pack(side='left')

        #Button(self.button_frame, text="NO CORRECT OPTION", padx=10, pady=5, font=("Helvetica", 10), command=self.unsuccessful_guess).pack(side='left')
        self.button_frame.place(x=self.canvas_width/2, y=300, anchor=CENTER)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PredictorApp()
    app.run()


#TODO: snygga till det, lägg till en argument till klassen där man kan välja modell (eller dylikt)