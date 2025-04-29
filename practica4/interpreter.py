""" Grammar:

;; Now our grammar could have multiple statements
program -> stmt | program

;; we support two statements
stmt -> assigment
      | print-stmt
assigment -> id '=' expr
print-stmt -> 'print' '(' expr ')'
expr -> term
      | term + term
      | term - term
  
term -> factor
      | factor * factor
      | factor / factor

factor -> digit | - digit
        | ( expr ) | - ( expr )
        | id | - id
        | base ^ exponent

base -> ( expr ) | digit
exponent -> ( expr ) | digit

digit -> [0-9] | digit
id -> [A-Z] | [a-z] | id
"""

# SymbolTable: Where we are going to store the id's or symbols
class SymbolTable:
    def __init__(self):
        self.table = {}
    
    def set(self, name, value):
        self.table[name] = value

    def exist(self, name):
        return name in self.table
    
    def get(self, name):
        try:
            return self.table[name]
        except KeyError:
            raise NameError(f"Variable {name} is not defined")

# Lexer: A simple class to pass from the stream of chars to tokens
class Lexer:
    # A few tokens, at the end the tokes are pairs of tuples
    EOF = ('EOF', 'EOF')
    PLUS = ('PLUS', '+')
    SUB = ('SUB', '-')
    MUL = ('MUL', '*')
    DIV = ('DIV', '/')
    POW = ('POW', '^')
    LPAREN = ('LPAREN', '(')
    RPAREN = ('RPAREN', ')')
    ASSIGN = ('ASSIGN', '=')
    PRINT = ('ID', 'print')
    # Default format of the ID's
    # ID = ('ID', '<name>')
    # NUMBER = ('NUMBER', int)
    
    def __init__(self, input_string, table: SymbolTable):
        self.input_string = input_string 
        self.tokens = []        # Will contain all the tokens
        self.index = 0
        self.table = table      # Catch the symbol table

        self.tokenize()         # Catch all the tokens

    def curr_char(self) -> str:
        return self.input_string[self.index]
    
    def move_next_char(self):
        self.index += 1
        
    def has_next_char(self) -> bool:
        return self.index < len(self.input_string)

    def tokenize_number(self) -> int:
        # tokenize a number of the kind `[0-9]`
        number = None
        while self.has_next_char() and self.curr_char().isdigit():
            if self.curr_char() == '0' and number == 0:
                raise Exception("Tokenizer: Invalid number")
            elif number is None:
                number = 0
            number *= 10
            number += int(self.curr_char())
            self.move_next_char()
        return number

    # tokenize_id: It will tokenize a `variable` and put it in the `Symbol Table`
    def tokenize_id(self) -> str:
        # it will tokenize an variable of `[a-z | A-Z]`
        variable = ""
        while self.has_next_char() and self.curr_char().isalpha():
            variable += self.curr_char()
            self.move_next_char()
        # Allocate the symbol with the value 'None'
        self.table.set(variable, None)
        return variable

    def tokenize(self):
        """
        Transform the actual stream of chars to a list of tuples which are tokens
        Example tokens: ('NUMBER', 3424), ('PLUS', '+'), ('EOF', 'EOF')
        """
        if self.input_string == "":
            raise Exception("Empty input string")

        while self.has_next_char():
            ch = self.curr_char()
            if ch.isdigit():
                self.tokens.append(('NUMBER', self.tokenize_number()))
                # NOTE: when it finished to valid the number
                # the current character is not a digit which means that
                # we need to `continue` here before executing again `next_char`
                continue
            elif ch.isalpha():
                self.tokens.append(('ID', self.tokenize_id()))
                # NOTE: When it finsihed to validate the variable
                # the current character is not an alpha which means that
                # we need to `continue` here before executing again `next_char`
                continue
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
            elif ch == '=':
                self.tokens.append(Lexer.ASSIGN)
            elif ch == '^':
                self.tokens.append(Lexer.POW)
            elif ch.isspace():
                # Is it is just space, like tabs or spaces just ignore
                pass
            else:
                # Return an error if 
                raise ValueError(f"Unknown char: {ch}") 
            self.move_next_char()
            
        # Adds the end of file token
        self.tokens.append(Lexer.EOF)

    def get_tokens(self):
        """ Return the list of tokens """
        return self.tokens




# Parser: is just the sintax analyzer
class Parser:
    def __init__(self, tokens, table):
        # Save the actual tokens from the lexer
        self.tokens = tokens
        self.table = table

        # Track the actual token
        self.position = 0
        self.current_token = None

        # Move to the first token
        self.next_token()

    def next_token(self):
        """ Moves to the move_next_char token, until reach the last token """
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

    
    def parse(self):
        """
        The main function that starts the interpreter
        """
        self.stmt() # First rule, of the grammar
        
        if self.current_token[0] != 'EOF':
            raise SyntaxError(f"Expect EOF, but found '{self.current_token[1]}'.")

    def stmt(self):
        """
        stmt -> id '=' expr
             |  'print' '(' ')'
             | stmt
        """
        while not self.match(Lexer.EOF):
            if self.match(Lexer.PRINT):
                self.next_token()
                if not self.match(Lexer.LPAREN):
                    raise SyntaxError("Expect an '(' after calling print")
                self.next_token()
                res = self.expr()     # Parse an expression
                if not self.match(Lexer.RPAREN):
                    raise SyntaxError("Expect an ')' after calling print")
                self.next_token()

                print(res)                 # Print res
            elif self.current_token[0] == "ID":
                variable = self.current_token[1]
                self.next_token()
                if not self.match(Lexer.ASSIGN):
                    raise SyntaxError(f"Expect '=' to define the variable {variable}")
                self.next_token()
                res = self.expr()
                # Save the interpreted value
                self.table.set(variable, res)
            else:
                raise SyntaxError(f"Expect PRINT or ID, but found {self.current_token[1]}")
            


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
        factor -> digit | - digit
                | ( expr ) | - ( expr )
                | id | - id
                | base ^ exponent | - base ^ exponent
        """
        factor = None
        negative = False
        if self.match(Lexer.SUB):
            negative = True
            self.next_token()

            
        if self.match(Lexer.LPAREN):
            # Do something here ...
            
            self.next_token()
            factor = self.expr()
            if not self.match(Lexer.RPAREN):
                raise SyntaxError(f"Expect to have ')' after having '(', but found {self.current_token[1]}")
            # Do something here ...
            
            self.next_token()
        elif self.current_token[0] == "NUMBER":
            factor = float(self.current_token[1])
            self.next_token()
        elif self.current_token[0] == "ID":
            variable = self.table.get(self.current_token[1])
            if variable is None:
                raise SyntaxError(f"Variable {self.current_token[1]} is not defined")
            factor = variable
            self.next_token()
        else:
            raise SyntaxError(f"Expect to have an integer number '[0-9]' or '(' with a new expression, but found '{self.current_token[1]}'")

        if self.match(Lexer.POW):
            self.next_token()
            exponent = None
            if self.match(Lexer.LPAREN):
                self.next_token()
                exponent = self.expr()
                if not self.match(Lexer.RPAREN):
                    raise SyntaxError(f"Expect to have ')' after having '(', but found {self.current_token[1]}")
                self.next_token()
            elif self.current_token[0] == "NUMBER":
                exponent = float(self.current_token[1])
                self.next_token()
            else:
                raise SyntaxError("Expected to have an exponent to be a number or an '( expr )'")
            factor = factor ** exponent
        
        if negative:
            return - factor
        return factor

def main():
    # A usage example
    
    # test_string = "  2 * ( 2 / ( 1 + 1))"
    # test_string = """

    
    # 22 + 2341 * (12 / (3 * 83)) - 1253

    
    # """
    # test_string = "  1 / 0"

    # test_string = """
    # x = 10 + 12
    # y = 23 + 33
    # z = x + y
    # print(z)
    # """
    # test_string = """
    # print((2 + 1) / 2)
    # """
    # Example program computing the chicharronera
    test_string = """
    a = 2
    b = 3
    c = - 6
    print((- b - ((b ^ 2 - 4 * a * c) ^ (1 / 2))) / (2 * a))
    print((- b + ((b ^ 2 - 4 * a * c) ^ (1 / 2))) / (2 * a))
    """
    
    
    try:
        # Create the Symbol Table
        table = SymbolTable()
        
        # Get the stream of tokens
        lexer = Lexer(test_string, table)
        tokens = lexer.get_tokens()
        # print("tokens: ", tokens)

        # Parse the tokens
        parser = Parser(tokens, table)
        parser.parse()

    except SyntaxError as e:
        print("Syntax Error:", e)
    except ValueError as e:
        print("Value Error:", e)

        
if __name__ == "__main__":
    main()
    
    


