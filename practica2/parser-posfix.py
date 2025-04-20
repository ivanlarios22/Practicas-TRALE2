
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
                self.tokens.append(self.PLUS)
            elif ch == '-':
                self.tokens.append(self.SUB)
            elif ch == '*':
                self.tokens.append(self.MUL)
            elif ch == '/':
                self.tokens.append(self.DIV)
            elif ch == '(':
                self.tokens.append(self.LPAREN)
            elif ch == ')':
                self.tokens.append(self.RPAREN)
            elif ch.isspace():
                # Is it is just space, like tabs or spaces just ignore
                i += 1
                continue
            else:
                # Return an error if 
                raise ValueError(f"Unknown char: {ch}") 
            i += 1
            
        # Adds the end of file token
        self.tokens.append(self.EOF)

    def get_tokens(self):
        """ Return the list of tokens """
        return self.tokens


# PostfixParser: is just the syntax analyzer, and returns the postfix notation 
class PostfixParser:
    PRECEDENCE = {
        Lexer.LPAREN: 0,
        Lexer.PLUS: 1, Lexer.SUB: 1,
        Lexer.MUL: 2, Lexer.DIV: 2
    }  # Define operator precedence, as map
    
    def __init__(self, tokens):
        # Save the actual tokens from the lexer
        self.tokens = tokens

        # Track the actual token
        self.position = 0
        self.current_token = None

        # Postfix necessary data structures
        self.postfix_expr = []  # The resultant expression
        self.postfix_stack_op = [] # The stack for managing the operators
        
        # Move to the first token
        self.next_token()

    def next_token(self):
        """ Moves to the next token, until reach the last token """
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
            self.position += 1
        else:
            self.current_token = Lexer.EOF
            
    def has_tokens(self) -> bool:
        """ Detects if there are more tokens """
        return self.position < len(self.tokens)

    def match(self, token : (str, str)) -> bool:
        """ Matches the current token with a given """
        return self.current_token[0] == token[0] and self.current_token[1] == token[1]
    
    def get_postfix(self) -> [tuple]:
        """ Returns the postfix array with tokens """
        return self.postfix_expr

    def get_postfix_str(self) -> str:
        """ Returns the postfix string expression """
        return " ".join(token[1] for token in self.postfix_expr)

    def handle_operator(self):
        """
        Manages operator precedence using a stack,

        while 'the last element in stack has higher precedence or hierarchy'; then
            'priorities those elements'
        """
        while (len(self.postfix_stack_op) > 0
               and self.PRECEDENCE[self.postfix_stack_op[-1]] >= self.PRECEDENCE[self.current_token]):
            self.postfix_expr.append(self.postfix_stack_op.pop())
        self.postfix_stack_op.append(self.current_token)

    def parse(self):
        """
        The main function that starts the syntax analizer
        """
        self.expr() # First rule, of the grammar
        
        while len(self.postfix_stack_op) > 0:
            self.postfix_expr.append(self.postfix_stack_op.pop())
        
        if self.current_token[0] != 'EOF':
            raise SyntaxError(f"Expect EOF, but found '{self.current_token[1]}'.")

    def expr(self):
        """
        expr -> term
              | term + term
              | term - term
        """
        self.term()     # First parse the digit
        
        while self.match(Lexer.PLUS) or self.match(Lexer.SUB):
            # Do something ...
            self.handle_operator()
            
            self.next_token()
            self.term()

    def term(self):
        """
        term -> factor
              | factor * factor
              | factor / factor
        """
        self.factor()

        while self.match(Lexer.MUL) or self.match(Lexer.DIV):
            # Do something ...
            self.handle_operator()
            
            self.next_token()
            self.factor()

    def factor(self):
        """
        factor -> digit
                | ( expr )
        """
        if self.match(Lexer.LPAREN):
            # Do something ...
            self.postfix_stack_op.append(Lexer.LPAREN)
            
            self.next_token()
            self.expr()
            
            if not self.match(Lexer.RPAREN):
                raise SyntaxError(f"Expect to have ')' after having '(', but found {self.current_token[1]}")

            # Do something ...
            
            while (len(self.postfix_stack_op) > 0 and self.postfix_stack_op[-1] != Lexer.LPAREN):
                self.postfix_expr.append(self.postfix_stack_op.pop())
                
            self.postfix_stack_op.pop() # Remove the LPAREN
            
            self.next_token()
        elif self.current_token[0] == 'DIGIT':
            self.postfix_expr.append(self.current_token)
            self.next_token()
        else:
            raise SyntaxError(f"Expect to have a digit '0, 1, 2, ..., 9' or '(' with a new expression, but found '{self.current_token[1]}'")

def main():
    # A usage example
    
    # test_string = "  2 * ( 2 / ( 1 + 1))"
    test_string = " 2 * (1 + (3 / ( 5 + 6)))"
    
    try:
        # Get the stream of tokens
        lexer = Lexer(test_string)
        tokens = lexer.get_tokens()
        print("tokens: ", tokens)

        # Parse the tokens
        parser = PostfixParser(tokens)
        parser.parse()
        
        print(f"{test_string} ->", parser.get_postfix_str())
    except SyntaxError as e:
        print("Syntax Error:", e)
    except ValueError as e:
        print("Value Error:", e)

        
if __name__ == "__main__":
    main()
