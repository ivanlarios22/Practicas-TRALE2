"""Grammar:
program -> 'begin' stmts 'end'
stmts -> stmt | stmts
stmt -> assig-stmt
      | print-stmt
      | if-else-stmt
      | while-stmt
      | declare-stmt
declare-stmt -> dtype id
assig-stmt -> dtype id '=' expr | id '=' expr
dtype -> 'int' | 'real' | 'bool'
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

# To fetch the arguments in the command line
import sys

# SymbolTable: Where we are going to store the id's or symbols

class SymbolTable:
    
    """
    '<val-name>' -> (val, dtype)
    """
    def __init__(self):
        self.table = {}
    
    def set(self, name, value = None):
        # If there is not value, then just register the name
        if value is not None:
            self.table[name] = (value, type(value))
        else:
            self.table[name] = (None, None)
        
    def set_type(self, name, dtype : type):
        """Sets the value as None"""
        self.table[name] = (None, dtype)

    def exist(self, name) -> bool:
        return name in self.table
    
    def get(self, name):
        try:
            return self.table[name][0]
        except KeyError:
            raise NameError(f"Variable {name} is not defined")

    def get_type(self, name) -> type:
        try:
            return self.table[name][1]
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
    POW = ('POW', '^')
    
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
    INTEGER = ('INTEGER', 'int')
    REAL = ('REAL', 'real')
    BOOLEAN = ('BOOLEAN', 'bool')
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
        INTEGER[1] : INTEGER,
        REAL[1] : REAL,
        BOOLEAN[1] : BOOLEAN,
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
        self.table.set(word)
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
            elif ch == '^':
                self.tokens.append(Lexer.POW)                
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


# Simple AST node classes
class ASTNode:
    pass

class Number(ASTNode):
    def __init__(self, value):
        self.value = value
        
    def __repr__(self):
        return f"Number({self.value})"

class Boolean(ASTNode):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Boolean({self.value})"
        

class Variable(ASTNode):
    def __init__(self, name : str, dtype : str):
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        return f"Variable('{self.name}')"

class BinOp(ASTNode):
    def __init__(self, op : tuple, left : ASTNode, right : ASTNode):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"BinOp('{self.op}', {self.left}, {self.right})"


class Op(ASTNode):
    def __init__(self, op : tuple, expr : ASTNode):
        self.op = op
        self.expr = expr
        
    def __repr__(self):
        return f"Op('{self.op}', {self.expr})"

class Assign(ASTNode):
    def __init__(self, target : Variable, expr : BinOp):
        self.target = target
        self.expr = expr

    def __repr__(self):
        return f"Assign({self.target}, {self.expr})"

class Print(ASTNode):
    def __init__(self, expr : BinOp):
        self.expr = expr

    def __repr__(self):
        return f"Print({self.expr})"

# A compound is a list of statements
class Compound(ASTNode):
    def __init__(self, statements : list[ASTNode]):
        self.statements = statements
        
    def __repr__(self):
        return f"Compound({self.statements})"    

class If(ASTNode):
    def __init__(self, condition : BinOp, then_body : Compound, else_body : Compound = None):
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body

    def __repr__(self):
        if self.else_body is not None:
            return f"If({self.condition}, {self.then_body}, {self.else_body})"
        return f"If({self.condition}, {self.then_body})"

# Usually a body is a compound of statements
class While(ASTNode):
    def __init__(self, condition : BinOp, body : Compound):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"While({self.condition}, {self.body})"



# Parser: is just the syntax analyzer, it returns an abstract syntax tree
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

        # Flags
        self.nested_if_stmt = False

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

    def get_dtype(self, token) -> type:
        """Based in the token we return the class object type"""
        if self.match(token, Lexer.INTEGER):
            return int
        elif self.match(token, Lexer.REAL):
            return float
        elif self.match(token, Lexer.BOOLEAN):
            return bool
        raise SyntaxError("Uknow data type")
            
    def match(self, token1 : (str, str), token2 : (str, str) = None) -> bool:
        """ Matches the current token with a given """
        if token2 is not None:
            return token1[0] == token2[0] and token1[1] == token2[1]
        return self.current_token[0] == token1[0] and self.current_token[1] == token1[1]

    def expect(self, token, msg = ""):
        if not self.match(token):
            raise SyntaxError(f"Expect '{token[0]}' {msg}, but found '{self.current_token[1]}'")

    def parse(self) -> ASTNode:
        """
        The main function that starts the syntax analizer
        """
        root = self.program() # First rule in the gramamar
        self.expect(Lexer.EOF, "We expect the end of the file")
        return root

    def program(self) -> Compound:
        """
        program -> 'begin' stmts 'end'
        """
        self.expect(Lexer.BEGIN, "to start a program")
        self.next_token()
        node_stmts = self.stmts()
        self.expect(Lexer.END, "to finish a program")
        self.next_token()
        
        return Compound(node_stmts)

    def stmts(self) -> list[ASTNode]:
        """
        stmts -> stmt | stmts
        """
        node_stmts = []
        node_stmts.append(self.stmt())
        while self.has_tokens() and not self.match(Lexer.END) and not self.match(Lexer.ELSE):
            node_stmts.append(self.stmt())
        return node_stmts

    def print_stmt(self) -> Print:
        self.next_token()
        node_expr = self.expr()
        return Print(node_expr)


    def if_stmt(self) -> If:
        # pdb.set_trace()
        self.next_token()
        node_cond = self.expr()
        node_then_body = Compound(self.stmts())
        node_else_body = None

        if self.match(Lexer.ELSE):
            self.next_token()
            if self.match(Lexer.IF):
                self.nested_if_stmt = True
            node_else_body = Compound(self.stmts())

        self.expect(Lexer.END, "for a if statement")
        if not self.nested_if_stmt:
            self.next_token()
        else:
            self.nested_if_stmt = False
                    
        return If(node_cond, node_then_body, node_else_body)

    def while_stmt(self) -> While:
        self.next_token()
        node_cond = self.expr()
        node_body = Compound(self.stmts())
        self.expect(Lexer.END, 'For a while loop')
        self.next_token()
        return While(node_cond, node_body)

    def assign_stmt(self, variable : Variable) -> Assign:
        if not self.match(Lexer.ASSIGN):
            raise SyntaxError("Expect '=' for assiging things")
        self.next_token()
        node_expr = self.expr()
        return Assign(variable, node_expr)

    def declare_stmt(self) -> Variable:
        dtype = self.get_dtype(self.current_token)
        self.next_token()
        var_name = self.current_token[1]
        
        if self.table.get_type(var_name) is not None:
            raise SyntaxError(f"Re defining the data type of var_name: {var_name}")
        self.table.set_type(var_name, dtype)
        self.next_token()
        return Variable(var_name, dtype)
            

    def stmt(self) -> ASTNode:
        """
        stmt -> assig-stmt
              | print-stmt
              | if-else-stmt
              | while-stmt
        """
        if self.match(Lexer.PRINT):
            return self.print_stmt()
            
        if self.match(Lexer.IF):
            return self.if_stmt()
        
        if self.match(Lexer.WHILE):
            return self.while_stmt()
            
        if self.match(Lexer.INTEGER) \
           or self.match(Lexer.REAL) \
           or self.match(Lexer.BOOLEAN):
            variable = self.declare_stmt()
            if self.match(Lexer.ASSIGN):
                return self.assign_stmt(variable)
            return variable
        if self.current_token[0] == "ID":
            var_name = self.current_token[1]
            self.table.get(var_name)
            dtype = self.table.get_type(var_name)
            if dtype is None:
                raise SyntaxError(f"Variable doesn't have a datatype defined {var_name}")
            variable = Variable(var_name, dtype)
            self.next_token()
            return self.assign_stmt(variable)

    def expr(self) -> BinOp or Op or ASTNode:
        """
        expr -> cond
              | cond '&&' cond
              | cond '||' cond
              | '!' cond
        """
        is_not_expr = False
        if self.match(Lexer.NOT):
            self.next_token()
            is_not_expr = True

        node = self.cond()
        while self.match(Lexer.OR) or self.match(Lexer.AND):
            # Do something here ...
            op = self.current_token
            self.next_token()
            right = self.cond()
            node = BinOp(op, node, right)
        if is_not_expr:
            return Op(Lexer.NOT, node)
        return node
    
    def cond(self) -> BinOp or ASTNode:
        """
        cond -> arith-expr
             | arith-expr '==' arith-expr
             | arith-expr '!='arith-expr
             | arith-expr '<' arith-expr
             | arith-expr '<=' arith-expr
             | arith-expr '>' arith-expr
             | arith-expr '>=' arith-expr
        """
        node = self.arith_expr()
        while self.match(Lexer.EQ) or self.match(Lexer.NQ) \
              or self.match(Lexer.GT) or self.match(Lexer.LT) \
              or self.match(Lexer.GE) or self.match(Lexer.LE):
            # Do something here ...
            op = self.current_token
            self.next_token()
            right = self.arith_expr()
            node = BinOp(op, node, right)
        return node
            
        
    def arith_expr(self):
        """
        arith-expr -> term
                   | term '+' term
                   | term '-' term
        """
        node = self.term()     # First parse the digit
        while self.match(Lexer.PLUS) or self.match(Lexer.SUB):
            # Do something here ...
            op = self.current_token
            self.next_token()
            right = self.term()
            node = BinOp(op, node, right)
        return node
            
    def term(self):
        """
        term -> factor
              | factor * factor
              | factor / factor
        """
        node = self.factor()
        while self.match(Lexer.MUL) or self.match(Lexer.DIV):
            # Do something here ...
            op = self.current_token
            self.next_token()
            right = self.factor()
            node = BinOp(op, node, right)
        return node
            
    def factor(self):
        """
        factor -> digit | - digit
                | ( expr ) | - ( expr )
                | id | - id
                | base ^ exponent | - base ^ exponent
        """
        factor_node = None
        negative = False
        if self.match(Lexer.SUB):
            negative = True
            self.next_token()
        
        if self.match(Lexer.LPAREN):
            self.next_token()
            factor_node = self.expr()
            self.expect(Lexer.RPAREN, "expect to have ')' after having '('")
            self.next_token()
        elif self.current_token[0] == 'NUMBER':
            val = float(self.current_token[1])
            self.next_token()
            factor_node = Number(val)
        elif self.match(Lexer.TRUE):
            self.next_token()
            factor_node = Boolean(True)
            
        elif self.match(Lexer.FALSE):
            self.next_token()
            factor_node = Boolean(False)
        elif self.current_token[0] == "ID":
            var_name = self.current_token[1]
            dtype = self.table.get_type(var_name)
            self.next_token()
            factor_node = Variable(var_name, dtype)
        else:
            raise SyntaxError(f"Expect to have an integer number '[0-9]' or '(' with a new expression, but found '{self.current_token[1]}'")
            
        if self.match(Lexer.POW):
            self.next_token()
            exponent_node = None
            if self.match(Lexer.LPAREN):
                self.next_token()
                exponent_node = self.expr()
                if not self.match(Lexer.RPAREN):
                    raise SyntaxError(f"Expect to have ')' after having '(', but found {self.current_token[1]}")
                self.next_token()
            elif self.current_token[0] == "NUMBER":
                exponent_node = Number(float(self.current_token[1]))
                self.next_token()
            else:
                raise SyntaxError("Expected to have an exponent to be a number or an '( expr )'")
            assert exponent_node is not None
            factor_node = BinOp(Lexer.POW, factor_node, exponent_node)

            

        if factor_node is None:
            raise SyntaxError(f"Expect to have a digit '0, 1, 2, ..., 9' or '(' with a new expression, but found '{self.current_token[1]}'")

        if negative:
            return Op(Lexer.SUB, factor_node)
        return factor_node



class CodeGenerator:
    def __init__(self, ast_root, table):
        self.ast_root = ast_root
        self.table = table
        self.code = []
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self) -> str:
        self.temp_count += 1
        return f"t{self.temp_count}"

    def new_label(self) -> str:
        self.label_count += 1
        return f"L{self.label_count}"

    def generate(self) -> list[str]:
        self.gen_node(self.ast_root)
        return self.code
    
    def gen_leaf(self, node : Variable or Number or Boolean):
        if isinstance(node, Variable):
            return node.name
        if isinstance(node, Number) or isinstance(node, Boolean):
            return node.value
        raise SyntaxError("Uknow leaf node")

    def gen_op(self, node : Op) -> str:
        temp = self.new_temp()
        self.code.append(f"\t{temp} = {node.op[1]} {self.gen_node(node.expr)}")
        return temp


    def gen_bin_op(self, node : BinOp) -> str:
        # expected_expr_type : type = None
        
        left = None
        right = None
        temp = self.new_temp()
        left = self.gen_node(node.left)
        right = self.gen_node(node.right)
        
        if isinstance(node, BinOp) and (node.op == Lexer.AND or node.op == Lexer.OR):
            assert left is bool or self.table.get_type(left) is bool
            assert right is bool or self.table.get_type(right) is bool
            self.table.set_type(temp, bool)
        elif isinstance(node, BinOp) and (node.op == Lexer.EQ or node.op == Lexer.NQ \
                                          or node.op == Lexer.GT or node.op == Lexer.GE \
                                          or node.op == Lexer.LT or node.op == Lexer.LE):
            if left is str and right is not str:
                assert self.table.get_type(left) == right
            elif right is str and left is not str:
                assert self.table.get_type(right) == left
            elif right is str and left is str:
                assert self.table.get_type(left) == self.table.get_type(right)
                
            self.table.set_type(temp, bool)
        elif isinstance(node, BinOp) and (node.op == Lexer.PLUS or node.op == Lexer.SUB \
                                          or node.op == Lexer.MUL or node.op == Lexer.DIV):
            self.table.set_type(temp, float)
        assert left is not None
        assert right is not None
        self.code.append(f"\t{temp} = {left} {node.op[1]} {right}")
        return temp

    def gen_assign(self, node : Assign) -> str:
        rhs = self.gen_node(node.expr)
        assert rhs is not None
        self.code.append(f"\t{node.target.name} = {rhs}")
        return node.target.name

    def gen_print(self, node : Print):
        temp = self.gen_node(node.expr)
        assert temp is not None
        self.code.append(f"\tprint {temp}")


    def gen_if(self, node : If):
        l_end = self.new_label()
        temp = self.gen_bin_op(node.condition)
        assert temp is not None
        if self.table.get_type(temp) is not bool:
            raise SyntaxError(f"Expect {temp} to be boolean")
        
        if node.else_body is not None:
            l_true = self.new_label()
            l_false = self.new_label()
            self.code.append(f"\tif {temp} goto {l_true}")
            self.code.append(f"\tifFalse {temp} goto {l_false}")
            self.code.append(f"{l_true}:")
            self.gen_node(node.then_body)
            self.code.append(f"\tjump {l_end}")            
            self.code.append(f"{l_false}:")
            self.gen_node(node.else_body)
        else:
            self.code.append(f"\tifFalse {temp} goto {l_end}")
            self.gen_node(node.then_body)
        self.code.append(f"{l_end}:")

    def gen_while(self, node : While):
        l_start = self.new_label()
        l_end = self.new_label()

        self.code.append(f"{l_start}:")
        temp = self.gen_node(node.condition)
        assert temp is not None
        if self.table.get_type(temp) is not bool:
            raise SyntaxError(f"Expect {temp} to be boolean")
        self.code.append(f"\tifFalse {temp} goto {l_end}")
        self.gen_node(node.body)
        self.code.append(f"\tjump {l_start}")
        self.code.append(f"{l_end}:")
            
    def gen_node(self, node : ASTNode) -> str or None:
        if isinstance(node, Compound):
            for node_stmt in node.statements:
                self.gen_node(node_stmt)
                
        elif isinstance(node, Assign):
            return self.gen_assign(node)

        elif isinstance(node, Print):
            self.gen_print(node)

        elif isinstance(node, BinOp):
            return self.gen_bin_op(node)

        elif isinstance(node, Variable) or isinstance(node, Number) or isinstance(node, Boolean):
            return self.gen_leaf(node)

        elif isinstance(node, If):
            self.gen_if(node)
            
        elif isinstance(node, While):
            self.gen_while(node)

        elif isinstance(node, Op):
            return self.gen_op(node)

    
def get_content(filename: str):
    """Get the content from the file"""
    try:
        with open(filename, "r") as f:
            return  f.read()
    except FileNotFoundError as e:
        print(f"Error! The file was not found: {e}")

def main():

    script_name = sys.argv[0]
    arguments = sys.argv[1:]
    if len(arguments) != 1:
        raise TypeError(f"This script requires at one argument which is the program. "
                        f"Usage: python {script_name} <program>")

    # Get the program
    program = get_content(arguments[0])

    # Create the table of contents
    table = SymbolTable()
    
    # Get the stream of tokens
    lexer = Lexer(program, table)
    tokens = lexer.get_tokens()

    # Parse the tokens and get the abstract syntax tree
    parser = Parser(tokens, table)
    ast_root = parser.parse()

    # From the abstarct syntax tree generate the three address code
    gen = CodeGenerator(ast_root, table)
    code = gen.generate()

    # Print the code
    for line in code:
        print(line)

        
if __name__ == "__main__":
    main()
    
    
