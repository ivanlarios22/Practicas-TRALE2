# Main Grammar:
# List -> RestOfList
# RestOfList -> + digit RestOfList
#            | e
# digit -> 0 | 1 | 2 | ... | 9


# Lexer: A simple class to pass from the stream of chars to tokens
class Lexer:
    def __init__(self, input_string):
        self.input_string = input_string 
        self.tokens = []        # Will contain all the tokens

        self.tokenize()         # Catch all the tokens

    def tokenize(self):
        """
        Transform the actual stream of chars to a list of tuples which are tokens
        Example tokens: ('DIGIT', '0'...'9'), ('PLUS', '+'), ('EOF', 'EOF')
        """
        
        s = self.input_string
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isdigit():
                self.tokens.append(('DIGIT', ch))
            elif ch == '+':
                self.tokens.append(('PLUS', ch))
            elif ch == '-':
                self.tokens.append(('SUB', ch))
            elif ch.isspace():
                # Is it is just space, like tabs or spaces just ignore
                i += 1
                continue
            else:
                # Return an error if 
                raise ValueError(f"Unknown char: {ch}") 
            i += 1
            
        # Adds the end of file token
        self.tokens.append(('EOF', 'EOF'))

    def get_tokens(self):
        """ Return the list of tokens """
        return self.tokens


# Parser: is just the sintax analyzer
class Parser:
    def __init__(self, tokens):
        # Save the actual tokens from the lexer
        self.tokens = tokens

        # Track the actual token
        self.position = 0
        self.current_token = None

        # Move to the first token
        self.next_token()

    def next_token(self):
        """ Moves to the next token, until reach the last token """
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
            self.position += 1
        else:
            self.current_token = ('EOF', 'EOF')

    def parse(self):
        """
        The main function that starts the syntax analizer
        """
        self.parse_List() # First rule, of the grammar
        
        if self.current_token[0] != 'EOF':
            raise SyntaxError("Stream of chars not valid for the grammar.")

    def parse_List(self):
        """
        List -> digit RestOfList
        """
        self.parse_digit()     # First parse the digit
        self.parse_RestOfList() # Then parse the rest of the list

    def parse_RestOfList(self):
        """
        RestOfList -> + digit RestOfList
                    | - digit RestOfList
                    | e
        """
        # Since we have a recursive production here
        # We need to fetch the plus token until get a different token
        while self.current_token[0] == 'PLUS' or self.current_token[0] == 'SUB':
            self.next_token()      # fetch '+' or '-'
            self.parse_digit()    # parse a digit
        # If there isn't another (+) token then it implies that there is lambda token (e)

    def parse_digit(self):
        """
        digit -> 0 | 1 | 2 | ... | 9
        """
        if self.current_token[0] == 'DIGIT':
            self.next_token()  # Parse a digit
        else:
            raise SyntaxError(f"We expect a digit but it found: {self.current_token}")


def main():
    # A usage example
    
    test_string = "2+8-5-2"
    
    try:
        # Get the stream of tokens
        lexer = Lexer(test_string)
        tokens = lexer.get_tokens()

        # Parse the tokens
        parser = Parser(tokens)
        parser.parse()
        
        print("Stream of chars accepted by the grammar.")

    except SyntaxError as e:
        print("Syntax Error:", e)
    except ValueError as e:
        print("Value Error:", e)

if __name__ == "__main__":
    main()
    
    
