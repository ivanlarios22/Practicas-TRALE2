
# Lexer: A simple class to pass from the stream of chars to tokens
class Lexer:
    # A few tokens, at the end the tokes are pairs of tuples
    EOF = ('EOF', 'EOF')
    PLUS = ('PLUS', '+')
    SUB = ('SUB', '-')
    MUL = ('MUL', '*')
    DIV = ('DIV', '/')
    LPAREN = ('LPAREN', '(')
    RPAREN = ('RPAREN', ')')
    
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
        if s == "":
            raise Exception("Empty input string")
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isdigit():
                self.tokens.append(('DIGIT', ch))
            elif ch == '+':
                self.tokens.append(Lexer.PLUS)
            elif ch == '-':
                self.tokens.append(Lexer.SUB)
            elif ch == '*':
                self.tokens.append(Lexer.MUL)
            elif ch == '/':
                self.tokens.append(Lexer.DIV)
            elif ch == '(':
                self.tokens.append(Lexer.LPAREN)
            elif ch == ')':
                self.tokens.append(Lexer.RPAREN)
            elif ch.isspace():
                # Is it is just space, like tabs or spaces just ignore
                i += 1
                continue
            else:
                # Return an error if 
                raise ValueError(f"Unknown char: {ch}") 
            i += 1
            
        # Adds the end of file token
        self.tokens.append(Lexer.EOF)

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
            self.current_token = Lexer.EOF
            
    def has_tokens(self):
        """ Detects if there are more tokens """
        return self.position < len(self.tokens)

    def match(self, token : (str, str)) -> bool:
        """ Matches the current token with a given """
        return self.current_token[0] == token[0] and self.current_token[1] == token[1]


    def parse(self) -> float:
        """
        The main function that starts the syntax analizer
        """
        digit = self.expr() # First rule, of the grammar

        # Do something here ...
        
        if self.current_token[0] != 'EOF':
            raise SyntaxError(f"Expect EOF, but found '{self.current_token[1]}'.")
        
        return digit

    def expr(self) -> float:
        """
        expr -> term
              | term + term
              | term - term
        """
        res = self.term()     # First parse the digit
        while self.match(Lexer.PLUS) or self.match(Lexer.SUB):
            op = self.current_token
            self.next_token()
            operand = self.term() # Get the second operand

            # Pre analisis
            if op[0] == Lexer.PLUS[0]:
                res += operand
            elif op[0] == Lexer.SUB[0]:
                res -= operand
        return res


    def term(self) -> float:
        """
        term -> factor
              | factor * factor
              | factor / factor
        """
        res = self.factor()

        while self.match(Lexer.MUL) or self.match(Lexer.DIV):
            op = self.current_token
            self.next_token()
            operand = self.factor() # Get the second operand

            # Pre analisis
            if op[0] == Lexer.MUL[0]:
                res *= operand
            elif op[0] == Lexer.DIV[0]:
                res /= operand
        return res

            
            


    def factor(self) -> float:
        """
        factor -> digit
                | ( expr )
        """

        factor = None
        if self.match(Lexer.LPAREN):
            # Do something here ...
            
            self.next_token()
            factor = self.expr()
            if not self.match(Lexer.RPAREN):
                raise SyntaxError(f"Expect to have ')' after having '(', but found {self.current_token[1]}")
            # Do something here ...
            
            self.next_token()
        elif self.current_token[0] == 'DIGIT':
            # Do something here ...
            factor = float(self.current_token[1])
            self.next_token()
        else:
            raise SyntaxError(f"Expect to have a digit '0, 1, 2, ..., 9' or '(' with a new expression, but found '{self.current_token[1]}'")
        return factor

def main():
    # A usage example
    
    # test_string = "  2 * ( 2 / ( 1 + 1))"
    # test_string = " 2 + 2 * (1 / (3 * 8)) - 1"
    test_string = "  1  / 0"
    
    try:
        # Get the stream of tokens
        lexer = Lexer(test_string)
        tokens = lexer.get_tokens()
        print("tokens: ", tokens)

        # Parse the tokens
        parser = Parser(tokens)
        print(parser.parse())

    except SyntaxError as e:
        print("Syntax Error:", e)
    except ValueError as e:
        print("Value Error:", e)

        
if __name__ == "__main__":
    main()
    
    


