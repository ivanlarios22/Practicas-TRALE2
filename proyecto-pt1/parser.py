"""Grammar:
program -> 'begin' stmts 'end'
stmts -> stmt | stmts
stmt -> assig-stmt
      | print-stmt
      | if-else-stmt
      | while-stmt
assig-stmt -> id '=' expr
print-stmt -> 'print' expr
if-else-stmt -> 'if' expr stmts 'endif'
             | 'if' expr stmts else-stmt
else-stmt -> 'else' stmts 'endif'
           | 'else' if-else-stmt
while-stmt -> 'while' expr stmts 'end'
expr -> cond
      | cond '&&' cond
      | cond '||' cond
      | '!' cond
cond -> arith-expr
      | arith-expr '==' arith-expr
      | arith-expr '!='arith-expr
      | arith-expr '<' arith-expr
      | arith-expr '<=' arith-expr
      | arith-expr '>' arith-expr
      | arith-expr '>=' arith-expr
arith-expr -> term
      | term '+' term
      | term '-' term
term -> factor
      | factor '*' factor
      | factor '/' factor
factor -> digit
        | '(' expr ')'
        | id
        | bool
digit -> [0-9] | digit
id -> [A-Z] | [a-z] | id
bool -> 'True' | 'False'
"""

# For debugging
import pdb

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

    # To print the symbol table
    def __str__(self) -> str:
        s = "{"
        for key, val in self.table.items():
            s += f"{key}: {val}, "
        s += "}"
        return s

    

# Lexer: A simple class to pass from the stream of chars to tokens
class Lexer:
    # A few tokens, at the end the tokes are pairs of tuples
    EOF = ('EOF', 'EOF')

    # Operations
    PLUS = ('PLUS', '+')
    SUB = ('SUB', '-')
    MUL = ('MUL', '*')
    DIV = ('DIV', '/')
    LPAREN = ('LPAREN', '(')
    RPAREN = ('RPAREN', ')')
    ASSIGN = ('ASSIGN', '=')
    NOT = ('NOT', '!')
    AND = ('AND', '&&')
    OR = ('OR', '||')
    EQ = ('EQ', '==')
    NQ = ('NQ', '!=')
    GT = ('GT', '<')
    GE = ('GE', '<=')
    LT = ('LT', '>')
    LE = ('LE', '>=')

    # Structure
    BEGIN = ('BEGIN', 'begin')
    END = ('END', 'end')
    IF = ('IF', 'if')
    ELSE = ('ELSE', 'else')
    WHILE = ('WHILE', 'while')
    PRINT = ('PRINT', 'print')

    
    # Data
    TRUE = ('TRUE', True)
    FALSE = ('FALSE', False)
    # Format for the numbers
    # NUMBER = ('NUMBER', int)
    # ID = ('ID', '<name>')

    # The key words of th language
    KEYWORDS = {
        BEGIN[1] : BEGIN,
        END[1] : END,
        IF[1] : IF,
        ELSE[1] : ELSE,
        WHILE[1] : WHILE,
        PRINT[1] : PRINT,
        TRUE[1] : TRUE,
        FALSE[1] : FALSE,
    }
    
    
    def __init__(self, input_string, table):
        self.table = table
        self.input_string = input_string
        self.index = 0
        self.tokens = []        # Will contain all the tokens
        self.tokenize()         # Catch all the tokens
        
    def has_next_char(self) -> bool:
        """Verify is there another character"""
        return self.index < len(self.input_string)

    def curr_char(self) -> str:
        """Gets the current character"""
        return self.input_string[self.index]
        
    def next_char(self):
        """Gets the next char"""
        self.index += 1

    def tokenize_number(self):
        # tokenize a number of the kind `[0-9]`
        number = None
        while self.has_next_char() and self.curr_char().isdigit():
            if self.curr_char() == '0' and number == 0:
                raise Exception("Tokenizer: Invalid number")
            elif number is None:
                number = 0
            number *= 10
            number += int(self.curr_char())
            self.next_char()
        self.tokens.append(('NUMBER', number))


    # tokenize_word: It will tokenize a `variable` and put it in the `Symbol Table`
    def tokenize_word(self):
        # it will tokenize an variable of `[a-z | A-Z]`
        word = ""
        while self.has_next_char() and self.curr_char().isalpha():
            word += self.curr_char()
            self.next_char()

        if word in Lexer.KEYWORDS:
            self.tokens.append(Lexer.KEYWORDS[word])
            return
        
        # Allocate the symbol with the value 'None'
        self.table.set(word, None)
        self.tokens.append(('ID', word))
        
    def tokenize(self):
        """
        Transform the actual stream of chars to a list of tuples which are tokens
        Example tokens: ('DIGIT', '0'...'9'), ('PLUS', '+'), ('EOF', 'EOF')
        """
        if self.input_string == "":
            raise Exception("Empty input string")
        
        while self.has_next_char():
            ch = self.curr_char()
            if ch.isdigit():
                self.tokenize_number()
                
                # NOTE: when it finished to valid the number
                # the current character is not a digit which means that
                # we need to `continue` here before executing again `next_char`
                continue
            elif ch.isalpha():
                self.tokenize_word()
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
            elif ch == '<':
                self.next_char()
                if self.curr_char() == '=':
                    self.tokens.append(Lexer.GE)
                else:
                    self.tokens.append(Lexer.GT)
                    continue # NOTE: Since we move the character we make continue
            elif ch == '>':
                self.next_char()
                if self.curr_char() == '=':
                    self.tokens.append(Lexer.LE)
                else:
                    self.tokens.append(Lexer.LT)
                    continue # NOTE: Since we move the character we make continue
            elif ch == '=':
                self.next_char()
                if self.curr_char() == '=':
                    self.tokens.append(Lexer.EQ)
                else:
                    self.tokens.append(Lexer.ASSIGN)
                    continue # NOTE: Since we move the character we make continue
            elif ch == '!':
                self.next_char()
                if self.curr_char() == '=':
                    self.tokens.append(Lexer.NQ)
                else:
                    self.tokens.append(Lexer.NOT)
                    continue # NOTE: Since we move the character we make continue
            elif ch == '&':
                if self.has_next_char():
                    self.next_char()
                    ch = self.curr_char()
                    if self.curr_char() == '&':
                        self.tokens.append(Lexer.AND)
                    else:
                        raise ValueError("Incomplete and operator '&&'") 
                else:
                    raise ValueError("Incomplete and operator '&&'") 
            elif ch == '|':
                if self.has_next_char():
                    self.next_char()
                    ch = self.curr_char()
                    if self.curr_char() == '|':
                        self.tokens.append(Lexer.OR)
                    else:
                        raise ValueError("Incomplete or operator '||'") 
                else:
                    raise ValueError("Incomplete or operator '||'") 
            elif ch.isspace():
                # Is it is just space, like tabs or spaces just ignore
                pass
            else:
                # Return an error if 
                raise ValueError(f"Unknown char: {ch}") 
            self.next_char()
            
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

    def parse(self):
        """
        The main function that starts the syntax analizer
        """
        self.program() # First rule in the gramamar
        
        if self.current_token[0] != 'EOF':
            raise SyntaxError(f"Expect EOF, but found '{self.current_token[1]}'.")

    def program(self):
        """
        program -> 'begin' stmts 'end'
        """
        if not self.match(Lexer.BEGIN):
            raise SyntaxError("Expect 'begin', to start the program")
        self.next_token()
        self.stmts()
        if not self.match(Lexer.END):
            raise SyntaxError("Expect 'end', to end a program")
        self.next_token()

    def stmts(self):
        """
        stmts -> stmt | stmts
        """
        self.stmt()
        while self.has_tokens() and not self.match(Lexer.END) and not self.match(Lexer.ELSE):
            self.stmt()

    def stmt(self):
        """
        stmt -> assig-stmt
              | print-stmt
              | if-else-stmt
              | while-stmt
        """
        if self.match(Lexer.PRINT):
            self.next_token()
            self.expr()
        elif self.match(Lexer.IF):
            self.next_token()
            self.expr()
            self.stmts()
            if self.match(Lexer.ELSE):
                self.next_token()
                self.stmts()
            if not self.match(Lexer.END):
                raise SyntaxError("Expect 'end' for a if statement")
            self.next_token()
        elif self.match(Lexer.WHILE):
            self.next_token()
            self.expr()
            self.stmts()
            if not self.match(Lexer.END):
                raise SyntaxError("Expect 'end' for a while loop")
            self.next_token()
        elif self.current_token[0] == 'ID':
            variable = self.table.get(self.current_token[1])
            self.next_token()
            if not self.match(Lexer.ASSIGN):
                raise SyntaxError("Expect '=' for assiging things")
            self.next_token()
            self.expr()

    def expr(self):
        """
        expr -> cond
              | cond '&&' cond
              | cond '||' cond
              | '!' cond
        """

        if self.match(Lexer.NOT):
            self.next_token()
            
            # Do something here ...

        self.cond()
        while self.match(Lexer.OR) or self.match(Lexer.AND):
            # Do something here ...
            
            self.next_token()
            self.cond()
    
    def cond(self):
        """
        cond -> arith-expr
             | arith-expr '==' arith-expr
             | arith-expr '!='arith-expr
             | arith-expr '<' arith-expr
             | arith-expr '<=' arith-expr
             | arith-expr '>' arith-expr
             | arith-expr '>=' arith-expr
        """
        self.arith_expr()
        while self.match(Lexer.EQ) or self.match(Lexer.NQ) \
              or self.match(Lexer.GT) or self.match(Lexer.LT) \
              or self.match(Lexer.GE) or self.match(Lexer.LE):
            # Do something here ...
            
            self.next_token()
            self.arith_expr()
            
        
    def arith_expr(self):
        """
        arith-expr -> term
                   | term '+' term
                   | term '-' term
        """
        self.term()     # First parse the digit
        while self.match(Lexer.PLUS) or self.match(Lexer.SUB):
            # Do something here ...
    
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
            # Do something here ...
            
            self.next_token()
            self.factor()
            
    def factor(self):
        """
        factor -> digit
                | '(' expr ')'
                | id
                | bool
        """
        if self.match(Lexer.LPAREN):
            self.next_token()
            self.expr()
            if not self.match(Lexer.RPAREN):
                raise SyntaxError(f"Expect to have ')' after having '(', but found {self.current_token[1]}")
            self.next_token()
        elif self.current_token[0] == 'NUMBER':
            factor = float(self.current_token[1])
            self.next_token()
        elif self.match(Lexer.TRUE):
            self.next_token()
        elif self.match(Lexer.FALSE):
            self.next_token()
        elif self.current_token[0] == "ID":
            variable = self.table.get(self.current_token[1])
            # Do something with the variable 
            # if variable is None:
            #     raise SyntaxError(f"Variable {self.current_token[1]} is not defined")
            self.next_token()
        else:
            raise SyntaxError(f"Expect to have a digit '0, 1, 2, ..., 9' or '(' with a new expression, but found '{self.current_token[1]}'")

def main():
    # A usage example
    
    # program = "  2 * ( 2 / ( 1 + 1))"
    # program = """
    # begin
    #    a = 1 + 2
    #    if a == 3
    #      print a
    #    end 
    # end
    # """
    program = """
    begin
        a = 0
        while a < 10
            a = a + 1
            if a == 5
                print a
            end 
        end
    end
    """
    
    
    try:
        table = SymbolTable()
        """
        tokens:  [('NUMBER', 2), ('PLUS', '+'), ('NUMBER', 2), ('MUL', '*'), ('LPAREN', '('), ('NUMBER', 2), ('DIV', '/'), ('NUMBER', 2), ('RPAREN', ')'), ('EOF', 'EOF')]
        """
        # Get the stream of tokens
        lexer = Lexer(program, table)
        tokens = lexer.get_tokens()
        print("tokens: ", tokens)
        
        print(table)
        # pdb.set_trace()
        
        # Parse the tokens
        parser = Parser(tokens, table)
        parser.parse()
        
        print(f"Stream of chars '{program}' accepted by the grammar.")
    except SyntaxError as e:
        print("Syntax Error:", e)
    except ValueError as e:
        print("Value Error:", e)

        
if __name__ == "__main__":
    main()
    
    
