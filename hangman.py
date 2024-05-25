import random

HANGMAN = [
    '--------------',
    '|            |',
    '|            |',
    '|            O',
    '|            |',
    '|           /|\\',
    '|            |',
    '|           / \\',
    ''
]

WORDS = ['CASA', 'CARRO', 'MONO', 'ESTERNOCLEIDOMASTOIDEO', 'PYTHON', 'DJANGO',
         'MILTON', 'LENIS', 'SWAPPS', 'LONGIA', 'UNITTESTING']

class Hangman():
    '''
    The Hangman game started...XXX
    '''

    def __init__(self, word_to_guess):
        self.failed_attempts = 0
        self.word_to_guess = word_to_guess
        self.game_progress = list('_' * len(self.word_to_guess))

    
    def find_indexing(self, letter):
        '''
        Method that takes a letter and return a list with his indexes 
        in the word to guess
        '''
        return [i for i, char in enumerate(self.word_to_guess) if letter == char]
    
    def is_invalid_letter(self, input_):
        '''
        Method to validate if an user input is not just a letter (it means the input
        is number or a text with more then 1 char)
        '''
        return input_.isdigit() or (input_.isalpha() and len(input_) > 1)
    
    def print_game_status(self):
        '''
        Method to print the word guess the blank spaces with the remaining attempts
        and guessed letter.
        '''
        print("\n")
        print("\n".join(HANGMAN[:self.failed_attempts]))
        print("\n")
        print(' '.join(self.game_progress))
    
    def update_progress(self, letter, indexes):
        '''
        Method to update game progress with guessed letter
        '''
        for index in indexes:
            self.game_progress[index] = letter

    def get_user_input(self):
        user_input = input("\nPlease type a letter: ")
        return user_input.upper()
    
    def play(self):
        '''
        method to play game
        '''
        while self.failed_attempts < len(HANGMAN):
            self.print_game_status()
            user_input = self.get_user_input()

            # check valid user input
            if self.is_invalid_letter(user_input):
                print("¡The input is not a letter!")
                continue
            # check user input is already not guessed
            if user_input in self.game_progress:
                print("You already gaussed that letter")
                continue

            if user_input in self.word_to_guess:
                indexes = self.find_indexing(user_input)
                self.update_progress(user_input, indexes)
                # if letter is not find in word
                if self.game_progress.count("_") == 0:
                    print("\n¡Yay! You win!")
                    print("The word is : {0}".format(self.word_to_guess))
                    quit()
            else:
                self.failed_attempts += 1
        print("\n¡OMG! You lost!")

if __name__ == "__main__":
    word_to_guess = random.choice(WORDS)
    hangman = Hangman(word_to_guess)
    hangman.play()