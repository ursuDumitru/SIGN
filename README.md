# SignLanguageRecognition

## TODO

- [+] make a way for the model to tell if there is no active sign detected
- [+] have 5 signs in total
- [?] build an app
- [+] create a csv files for sign labels that I train and use
- [+] restructure the project (use snake_case)
- [+] make the model be built on static images rather than video sequences
      (create a mode where you have the camera+mediapipe opened and you can press a button to take a picture)
- [?] create tests for the project
- [+] think if you will use x,y,z or just x,y (also, for normalization which will be the max value ?)
- [] *plot* the data
- [+] *plot* validation and training loss
- [+] create more data(split_size -> 0.8/0.2 or 0.75/0.25, the more data the bigger traing size)
- [+] try different batch sizes
- [+] move csv files to a separate 'data' folder
- [+] make a mode in which a point appears in the frame and I can move it with gesture (up, down, left, right)
- [+] THINK: maybe I need to overfit my model, because Sign language detection must be precise not ambiguous
- [+] pass variables as function arguments, keep class params that are only specific to the respective class
- [+] more training data for abc
- [+] more layers in the model
- [+] make it only one hand
- [+] insert letters has a delay
- [+] right hand for word/letters, left hands for picking actions # work fine without defining handedness
- [] 500 images per sign ???
- [] translate the sentence in a language of choice
- [] limit the word length
- [] limit the sentence length
- [] add a voice option to the translated text
- [] write comments above each function
- [] MAKE A MODEL FOR DYNAMIC SIGNS !!!!!!!!!!!!!!!!!

## TODO Training

- [] new accurate data-set
- [] create tests for the model
- [] test every code branch so that it works as expected

(maybe have like 2 examples of use for my license, one to create sentences, another to control a presentation, or a mini game like google dinosaur)

## TODO Learning

- [] centroids
- [] learn about *layers*
- [+] learn about *input*, *how information goes through the layers*, *output*
- [] learn about *optimizers*, *loss* and *metrics*
- [] learn about *Sequential*
- [] learn about *random_state*
- [] learn about *Dropout*
- [] learn about *confusion matrix*

## Good to know

- Alt + Shift + Up/Down = move lines