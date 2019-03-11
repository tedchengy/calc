"""
feature engineering
使用脚本的方法创建新变量
"""
import sys
import numpy as np
import pandas as pd
import re
from sly import Lexer, Parser
from itertools import combinations


class CalcLexer(Lexer):
    tokens = {ID, FUNC, NUMBER, RANGE, PLUS, TIMES, MINUS, DIVIDE, LPAREN, RPAREN, SPLIT, }
    ignore = ' \t'

    # Tokens
    ID = r'@[A-Za-z0-9_:\u4E00-\u9FA5]+'
    # re.findall(r'@[A-Za-z0-9_:\u4E00-\u9FA5]+','@var+@var_123_1+@var:123:1_1:a')
    FUNC = r'[A-Za-z]+[0-9]*_?[A-Za-z0-9_]*'
    RANGE = r'\[[0-9,]+\]'
    NUMBER = r'\d+'

    # Special symbols
    PLUS = r'\+'
    MINUS = r'-'
    TIMES = r'\*'
    DIVIDE = r'/'
    LPAREN = r'\('
    RPAREN = r'\)'
    SPLIT = r'\,'
#    ASSIGN = r'='

    # Ignored pattern
    ignore_newline = r'\n+'

    # Extra action for newlines
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        self.index += 1


class CalcParser(Parser):
    tokens = CalcLexer.tokens

    precedence = (
        ('left', SPLIT),
        ('left', PLUS, MINUS),
        ('left', TIMES, DIVIDE),
        ('left', FUNC),
    )

    def __init__(self):
        self.funcs = {'exp': self._py_exp,
                      'min': self._py_min,
                      'max': self._py_max,
                      'sum': self._py_sum,
                      'mean': self._py_mean,
                      'eq': self._py_eq,
                      'log2': self._py_log2,
                      'log10': self._py_log10,
                      'ln': self._py_ln,
                      'log': self._py_log,
                      'pow': self._py_pow,

                      'Concat': self._py_concat,
                      'Diff': self._py_diff,
                      'Accumulate': self._py_acc,
                      'Quantile_bin': self._py_quantile_bin_0,
                      'Quantile_bin_1': self._py_quantile_bin_1,
                      'Custom_Quantile_bin': self._py_custom_quantile_bin,
                      }

    def _py_exp(self, expr):
        return np.exp(expr)

    def _py_min(self, expr):
        if type(expr) == pd.core.frame.DataFrame:
            return np.min(expr,axis=1)
        else:
            return np.min(np.min(expr),)

    def _py_max(self, expr):
        if type(expr) == pd.core.frame.DataFrame:
            return np.max(expr,axis=1)
        else:
            return np.max(np.max(expr),)

    def _py_sum(self, expr):
        if type(expr) == pd.core.frame.DataFrame:
            return np.sum(expr,axis=1)
        else:
            return np.sum(np.sum(expr),)

    def _py_mean(self, expr):
        if type(expr) == pd.core.frame.DataFrame:
            return np.mean(expr,axis=1)
        else:
            return np.mean(np.mean(expr),)        

    def _py_eq(self, expr):
        # 暂时equal只支持双目运算
        if type(expr) == pd.core.frame.DataFrame:
            return expr.iloc[:, 0] == expr.iloc[:, 1]
        else:
            raise Exception("Equal operation is wrong, please check the expression")

    def _py_log2(self, expr):
        # RuntimeWarning: invalid value encountered
        return np.log2(expr)

    def _py_log10(self, expr):
        # RuntimeWarning: invalid value encountered
        return np.log10(expr)

    def _py_ln(self, expr):
        # RuntimeWarning: invalid value encountered
        return np.log(expr)

    def _py_log(self, expr):
        # RuntimeWarning: invalid value encountered
        # log只支持双目运算
        if type(expr) == pd.core.frame.DataFrame and len(expr.columns) == 2:
            return np.log(expr.iloc[:, 0]) / np.log(expr.iloc[:, 1])
        else:
            raise Exception("log operation is wrong, please check the expression")

    def _py_pow(self, expr):
        # pow只支持双目运算
        if type(expr) == pd.core.frame.DataFrame and len(expr.columns) == 2:
            return np.power(expr.iloc[:, 0], expr.iloc[:, 1])
        else:
            raise Exception("log operation is wrong, please check the expression")

    def _py_concat(self, expr):
        if type(expr) != pd.core.frame.DataFrame:
            raise Exception("concat operation is wrong, please check the expression")
    
        p = expr.iloc[:, -1]
        var = expr.iloc[:, :-1]
        p0 = int(p[0])
        for i in range(p_rows):
            if p[i] != p0:
                raise Exception("concat operation is wrong, please check the expression")
        if p0 < 2 or p0 > len(var.columns):
            raise Exception("concat p parameter, please check the expression")
    
        rs = pd.DataFrame()
        for cs in combinations(range(len(var.columns)), p0):
            vs = [var.iloc[:, c] for c in cs]
            r = pd.concat(vs, axis=1).apply(lambda x: '_'.join(np.array(x).astype("str")), axis=1)
            rs = pd.concat([rs, r], axis=1)
        return rs

    def _py_diff(self, expr):
        if type(expr) != pd.core.frame.DataFrame:
            raise Exception("diff operation is wrong, please check the expression")

        p = expr.iloc[:, -1]
        var = expr.iloc[:, :-1]
        p0 = int(p[0])
        for i in range(p_rows):
            if p[i] != p0:
                raise Exception("diff operation is wrong, please check the expression")
        return var.diff(p0)

    def _py_acc(self, expr):
        return expr.cumsum()

    def _py_quantile_bin_0(self, expr):
        # 暂只支持双目
        if type(expr) != pd.core.frame.DataFrame:
            raise Exception("quantile_bin operation is wrong, please check the expression")

        b = expr.iloc[:, -1]
        var = expr.iloc[:, 0]
        bins = int(b[0])
        for i in range(p_rows):
            if b[i] != bins:
                raise Exception("quantile_bin operation is wrong, please check the expression")
        return pd.cut(var, bins, right=True, include_lowest=True)

    def _py_quantile_bin_1(self, expr):
        # 暂只支持双目
        if type(expr) != pd.core.frame.DataFrame:
            raise Exception("quantile_bin operation is wrong, please check the expression")

        b = expr.iloc[:, -1]
        var = expr.iloc[:, 0]
        bins = int(b[0])
        for i in range(p_rows):
            if b[i] != bins:
                raise Exception("quantile_bin operation is wrong, please check the expression")
        return pd.qcut(var, bins, duplicates='drop')

    def _py_custom_quantile_bin(self, expr):
        # 暂只支持双目
        if type(expr) != pd.core.frame.DataFrame:
            raise Exception("custom_quantile_bin operation is wrong, please check the expression")
        r = expr.iloc[:, -1]
        var = expr.iloc[:, 0]
        ra = r[0]
        for i in range(p_rows):
            if r[i] != ra:
                raise Exception("custom_quantile_bin operation is wrong, please check the expression")
        min_value = var.min()
        max_value = var.max()
        ra = np.clip(ra, min_value, max_value)
        ra = sorted(set(ra))
        if ra[0] > min_value:
            ra.insert(0, min_value)
        if ra[-1] < max_value:
            ra.insert(len(ra), max_value)
        if len(ra) < 2:
            ra = [min_value, max_value]
        return pd.cut(var, ra, right=True, include_lowest=True)

    @_('expr')
    def statement(self, p):
        print(p.expr)
        return p.expr

    @_('expr PLUS expr')
    def expr(self, p):
        return p.expr0 + p.expr1

    @_('expr MINUS expr')
    def expr(self, p):
        return p.expr0 - p.expr1

    @_('expr TIMES expr')
    def expr(self, p):
        return p.expr0 * p.expr1

    @_('expr DIVIDE expr')
    def expr(self, p):
        return p.expr0 / p.expr1

    @_('MINUS expr')
    def expr(self, p):
        return -p.expr

    @_('LPAREN expr RPAREN')
    def expr(self, p):
        return p.expr

    @_('expr SPLIT expr')
    def expr(self, p):
#        print(type(p.expr0), type(p.expr1))
#        print(p.expr1)
        if type(p.expr0) == str:
            p.expr0 = eval(p.expr0)
        if type(p.expr1) == str:
            p.expr1 = eval(p.expr1)

        if isinstance(p.expr0,(int,float,list)):
            p.expr0 = pd.Series([p.expr0] * p_rows)
        if isinstance(p.expr1,(int,float,list)):
            p.expr1 = pd.Series([p.expr1] * p_rows)
        return pd.concat([p.expr0, p.expr1], axis=1)

    @_('NUMBER')
    def expr(self, p):
        return int(p.NUMBER)

    @_('FUNC expr')
    def expr(self, p):
#                print(p.FUNC)
#                print(self.funcs[p.FUNC])
#                print(p.expr)
        return self.funcs[p.FUNC](p.expr)

    @_('ID')
    def expr(self, p):
        p_ID = p.ID[1:]
        return p_df[p_ID]

    @_('RANGE')
    def expr(self, p):
        return p.RANGE


if __name__ == '__main__':
    p_df = pd.DataFrame(np.random.randn(100, 6) * 100,
                        columns=['var', '123', 'var_123', 'var:123', 'var+123', 'var 123'])
    p_rows = len(p_df)

    lexer = CalcLexer()
    parser = CalcParser()

    sa = '-10+(5+2)*5/4'  # -1.25
    sb = '- 3 + 5 * exp(10+20)'  # 53432372907619.31
    sc = 'min(5,1*10,1)/5'  # 0.2
    sd = 'exp(max(5,1*10,1)/5)'  # 7.38905609893065

    se = '@var+@123+@var_123'  # p_df['var']+p_df['123']+p_df['var_123']
    sf = 'exp(@var+@123+@var_123+@var:123)'  # np.exp(p_df['var']+p_df['123']+p_df['var_123']+p_df['var:123'])
    sg = 'min(@var,@123+@var_123)'  # np.min(pd.concat([p_df['var'],p_df['123']+p_df['var_123']],axis=1))
    sh = 'min(exp(@var+1),@123)'
    si = 'eq(@var,@123)'
    sj = 'log10(@var)'
    sj = 'mean(@var,10)'
    sk = 'log(@var,10)'
    sl = 'pow(@var,2)'

    sm = 'Concat(@var,@123,@var_123,2)'#返回三列
    sn = 'Diff(@var,1)'
    so = 'Accumulate(@var)+@var'
    sp = 'Quantile_bin(@var,3)'
    st = 'Custom_Quantile_bin(@var,[0,10])'
    
    a1='exp(@var+1)'
    a2='ln(exp(@var))'
    a3='min(exp(@var),@123)'
    a4='Concat(Concat(@var,@123,@var_123,2),2)'
    a5='Accumulate(@var)+1'
    a6='Accumulate(@var,@var_123)+1'
    a7='Concat(Accumulate(@var,@var_123),2)'
    
     
    b1='Concat(Concat(@var,@123,@var_123,2),2)'
    b2='Accumulate(@var)+1'
    
    for tok in lexer.tokenize(sh):
        print(tok)
    new_df=parser.parse(lexer.tokenize(sh))
    #new_name=['new_name1','new_name2','new_name3']
    new_name=['new_name']
    
    
    if type(new_df)==pd.core.series.Series:
        new_df.name=new_name[0]
    elif isinstance(new_df,(int,float,np.int64,np.float64)):
        new_df=pd.Series([new_df] * p_rows)
        new_df.name=new_name
    elif type(new_df)==pd.core.frame.DataFrame:
        if len(new_df.columns)==len(new_name):
            new_df.columns=new_name
        else:
            raise Exception("new_variable_name_columns is wrong, please check the new variable name")
    
    result_df=pd.concat([p_df,new_df],axis=1)
    print(result_df.iloc[0:20])
    
    
for cs in combinations(range(len(var.columns)), p0):
    vs = [var.iloc[:, c] for c in cs]
    r = pd.concat(vs, axis=1).apply(lambda x: '_'.join(np.array(x).astype("str")), axis=1)
    rs = pd.concat([rs, r], axis=1)
    
    