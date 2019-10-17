#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import typing
import operator
import functools

import lark


class ExpressionError(Exception):
    pass


class ExpressionSyntaxError(ExpressionError):
    pass


class ExpressionNameError(ExpressionError):
    pass


class Function(typing.NamedTuple):
    name: str
    args: tuple

    def __repr__(self):
        return '%s(%s)' % (self.name, ', '.join(map(repr, self.args)))

    def to_dict(self):
        args = []
        for arg in self.args:
            if isinstance(arg, Function):
                args.append(arg.to_dict())
            else:
                args.append(arg)
        return {'name': self.name, 'args': args}

    @classmethod
    def from_dict(cls, d):
        args = tuple(cls.from_dict(x) if isinstance(x, dict) else x
                     for x in d['args'])
        return cls(d['name'], args)


class CallGraphGenerator(lark.Transformer):

    def __getattr__(self, item):
        if item.startswith('expr_'):
            return self._convert_expr
        raise AttributeError

    def name(self, args):
        return Function('_name', (str(args[0]),))

    def string(self, args):
        quote = args[0][0]
        return args[0].replace('\\' + quote, quote).replace('\\\\', '\\')[1:-1]

    def number(self, args):
        tok = args[0]
        if tok.type == 'FLOAT_NUMBER':
            return float(tok)
        else:
            return int(tok, 0)

    def const(self, args):
        return Function('_const', (tuple(map(str, args[0].children)),))

    def assign_stmt(self, args):
        # (_name, Token('='), val)
        return Function('=', (args[0].args[0], args[2]))

    def expr1_diff(self, args):
        return Function("'", (args[0],))

    def expr1_unary(self, args):
        return Function('u' + str(args[0]), (args[1],))

    def _convert_expr(self, args):
        """
        Merge same operations.
        eg. 1+2+3-4-5+6+7 -> +(-(+(1, 2, 3), 4, 5), 6, 7)
        """
        left = args[0]
        it = iter(args[1:])
        lastfn = None
        lastargs = []
        for middle, right in zip(it, it):
            fn = str(middle)
            if fn == lastfn:
                lastargs.append(right)
            else:
                if lastfn:
                    left = Function(lastfn, tuple(lastargs))
                lastargs = [left, right]
            lastfn = fn
        if lastfn:
            left = Function(lastfn, tuple(lastargs))
        return left

    def funccall(self, args):
        fn = '.'.join(args[0].children)
        if len(args) > 1:
            pargs = tuple(args[1].children)
        else:
            pargs = ()
        return Function(fn, pargs)

    def getitem(self, args):
        return Function('_getitem', (args[0], tuple(args[1].children)))

    def simple_stmt(self, args):
        return Function('_stmts', tuple(args))

    def suite(self, args):
        stmts = []
        for arg in args:
            if arg[0] == '_stmts':
                stmts.extend(arg[1])
            else:
                stmts.append(arg)
        return Function('_stmts', tuple(stmts))

    def start(self, args):
        return self.suite(args)

    def if_stmt(self, args):
        if_blocks = []
        else_block = None
        for arg in args:
            if arg.data == 'if_else_block':
                else_block = arg.children[0]
            else:
                if_blocks.append(tuple(arg.children))
        return Function('_if', (tuple(if_blocks), (else_block,)))


_graph_generator = CallGraphGenerator()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'expr.lark'), 'r', encoding='utf-8') as f:
    _grammar_file = f.read()

parser_expr = lark.Lark(
    _grammar_file, start="start_expr", parser="lalr",
    transformer=_graph_generator)
parser_file = lark.Lark(
    _grammar_file, start="start", parser="lalr", transformer=_graph_generator)

del _grammar_file


def expr_reduce(fn):
    @functools.wraps(fn)
    def wrapped(*args):
        return functools.reduce(fn, args)
    return wrapped


class Expression:

    functions = {
        '_': (lambda x: x),
        #"'": left_diff1d,
        '^': expr_reduce(math.pow),
        'u+': operator.pos,
        'u-': operator.neg,
        'u!': operator.inv,
        '+': expr_reduce(operator.add),
        '-': expr_reduce(operator.sub),
        '*': expr_reduce(operator.mul),
        '/': expr_reduce(operator.truediv),
        '%': expr_reduce(operator.mod),
        '//': expr_reduce(operator.floordiv),
        '<<': expr_reduce(operator.lshift),
        '>>': expr_reduce(operator.rshift),
        '<': expr_reduce(operator.lt),
        '<=': expr_reduce(operator.le),
        '==': expr_reduce(operator.eq),
        '!=': expr_reduce(operator.ne),
        '>=': expr_reduce(operator.ge),
        '>': expr_reduce(operator.gt),
        '&': expr_reduce(operator.and_),
        '|': expr_reduce(operator.or_),
        '&&': expr_reduce(lambda x, y: x and y),
        '||': expr_reduce(lambda x, y: x or y),
        "int": int,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
    }
    consts = {
        'pi': math.pi,
        'e': math.e,
    }

    def __init__(self, expr, multi_expr=False):
        self.local_functions = {
            '=': self.assign,
            '_name': self.get_name,
            '_const': self.get_const,
            '_getitem': operator.getitem,
            '_stmts': self.evaluate_stmts,
            '_if': self.evaluate_if,
        }
        if isinstance(expr, Function):
            self.tree = expr
        else:
            try:
                if multi_expr:
                    self.tree = parser_file.parse(expr.strip() + '\n')
                else:
                    self.tree = parser_expr.parse(expr.strip())
            except lark.UnexpectedInput as ex:
                raise ExpressionSyntaxError(
                    "syntax error at line %d column %d" % (ex.line, ex.column))
            # literal value
            if not isinstance(self.tree, Function):
                self.tree = Function('_', (self.tree,))

        self.vars = {}
        self.local_consts = {}

    def __repr__(self):
        return 'Expression(%r)' % (self.tree,)

    @classmethod
    def from_dict(cls, d):
        return cls(Function.from_dict(d))

    def to_dict(self):
        return self.tree.to_dict()

    def assign(self, name, val):
        self.vars[name] = val

    def get_name(self, name):
        try:
            return self.vars[name]
        except KeyError:
            pass
        try:
            return self.consts[name]
        except KeyError:
            raise ExpressionNameError('variable "%s" is not defined' % name)

    def get_const(self, name):
        # config value: {a.b.c}
        return self.local_consts[name]

    def dependent_vars(self, root=None):
        if root is None:
            root = self.tree
        depvars = set()
        for arg in root.args:
            if isinstance(arg, Function):
                if arg.name == '_name':
                    if arg.args[0] not in self.consts:
                        depvars.add(arg.args[0])
                else:
                    depvars.update(self.dependent_vars(arg))
        return depvars

    def assigned_vars(self, root=None):
        if root is None:
            root = self.tree
        assigned = set()
        for arg in root.args:
            if isinstance(arg, Function):
                if arg.name == '=':
                    assigned.add(arg.args[0])
                else:
                    assigned.update(self.assigned_vars(arg))
        return assigned

    def evaluate(self, variables=None, consts=None):
        if variables:
            self.vars = variables
        if consts:
            self.local_consts = consts
        return self.evaluate_fn(self.tree)

    def __call__(self, variables=None, consts=None):
        return self.evaluate(variables, consts)

    def evaluate_fn(self, expr):
        if not isinstance(expr, Function):
            return expr
        fn = self.local_functions.get(expr.name, self.functions.get(expr.name))
        if fn is None:
            raise ExpressionNameError('function "%s" not found' % expr.name)
        fnargs = list(map(self.evaluate_fn, expr.args))
        return fn(*fnargs)

    def evaluate_if(self, if_blocks, else_block):
        for test, stmts in if_blocks:
            # if, elseif
            if self.evaluate_fn(test):
                self.evaluate_fn(stmts)
                break
        else:
            # else
            if else_block:
                self.evaluate_fn(else_block)

    def evaluate_stmts(self, *stmts):
        for st in stmts:
            self.evaluate_fn(st)


if __name__ == '__main__':
    while True:
        s = input('> ')
        if s:
            if '@' in s:
                s = s.replace('@', '\n')
                multi = True
            else:
                multi = False
            try:
                expr = Expression(s, multi)
                print(expr.tree)
                #print(expr.to_dict())
                assert Function.from_dict(expr.to_dict()) == expr.tree
                print(expr.dependent_vars())
                print(expr.assigned_vars())
                print(expr())
                if expr.vars:
                    print(expr.vars)
            except Exception as ex:
                print(repr(ex))
        else:
            break
