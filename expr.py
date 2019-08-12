#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyparsing as pp

pp.ParserElement.enablePackrat()

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def inf1swap(toks):
    left = toks[0][0]
    for right in toks[0][1:]:
        left = [right, left]
    return [left]

def inf2swap(toks):
    left = toks[0][0]
    for middle, right in pairwise(toks[0][1:]):
        left = [middle, left, right]
    return [left]

class BaseExpression:

    arith_def = [
        (pp.oneOf("!"), 1, pp.opAssoc.LEFT, inf1swap),
        (pp.oneOf('+ - ~'), 1, pp.opAssoc.RIGHT),
        (pp.oneOf('* / %'), 2, pp.opAssoc.LEFT, inf2swap),
        (pp.oneOf('+ -'), 2, pp.opAssoc.LEFT, inf2swap),
        (pp.oneOf('<< >>'), 2, pp.opAssoc.LEFT, inf2swap),
        (pp.oneOf('&'), 2, pp.opAssoc.LEFT, inf2swap),
        (pp.oneOf('^'), 2, pp.opAssoc.LEFT, inf2swap),
        (pp.oneOf('|'), 2, pp.opAssoc.LEFT, inf2swap),
        (pp.oneOf('< <= == != >= >'), 2, pp.opAssoc.LEFT, inf2swap),
    ]
    operators = {}
    functions = {}
    consts = {}

    def __init__(self, s):
        num_int = pp.Regex(r"[+-]?[0-9]+").setParseAction(lambda toks: int(toks[0]))
        num_hex = pp.Regex(r"0x[0-9a-fA-F]+").setParseAction(
            lambda toks: int(toks[0], 0))
        num_float = pp.Regex(
            r"[+-]?[0-9]*\.?[0-9]+(:?[eE][+-]?[0-9]+)?").setParseAction(
            lambda toks: float(toks[0]))
        num = num_int ^ num_hex ^ num_float
        identifier = pp.Regex(r"[A-Za-z_][A-Za-z0-9_.]*")
        functor = identifier
        varname = identifier.setParseAction(self.add_var)

        lparen = pp.Literal("(").suppress()
        rparen = pp.Literal(")").suppress()

        expr = pp.Forward()
        func_args = pp.delimitedList(expr, ',')
        func_call = pp.Group(functor + lparen + pp.Optional(func_args) + rparen)
        atom = func_call | num | varname
        arith_expr = pp.infixNotation(atom, self.arith_def)
        expr <<= arith_expr

        self.expr = expr
        self.vars = set()
        self.parsed = expr.parseString(s, parseAll=True)

    def add_var(self, toks):
        self.vars.add(toks[0])

    def evaluate(self):
        raise NotImplementedError

if __name__ == '__main__':
    s = input('> ')
    exp = BaseExpression(s)
    print(exp.parsed)
    print(exp.vars)
