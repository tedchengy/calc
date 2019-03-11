"""
feature engineering
使用脚本的方法创建新变量
修改完本脚本后事先执行下main函数
"""
import collections
import re
import traceback
from itertools import combinations

import numpy as np
import pandas as pd

Token = collections.namedtuple('Token', ['type', 'value', 'line', 'column'])
BaseType = (int, float, np.int32, np.int64, np.float32, np.float64,)


def de_func(f):
    """
    使用一个函数包装器，把pd.Series变量放到list中
    :param f: 
    :return: 
    """

    def de_param(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], (list, BaseType,)):
            rs = f(self, args[0], *args[1:], **kwargs)
        else:
            index = -1
            for i, s in enumerate(args):
                if isinstance(s, (pd.DataFrame, pd.Series, np.ndarray)):
                    index = i
                else:
                    break
            if index >= 0:
                if 0 == index and isinstance(args[:index + 1][0], pd.DataFrame):
                    a = args[:index + 1][0]
                    rs = f(self, [a[i] for i in a], *args[index + 1:], **kwargs)
                else:
                    rs = f(self, args[:index + 1], *args[index + 1:], **kwargs)
            else:
                raise Exception('syntax error')

        if isinstance(rs, BaseType):
            return rs

        if len(rs) == 1:
            return rs[0]

        if isinstance(rs, list):
            rs = pd.concat(rs, axis=1)

        return rs

    return de_param


class TransformScripts:
    function_type_1 = ' ,log10,ln,exp'.split(',')  # 单目运算
    function_type_2 = 'log,pow,eq'.split(',')  # 二元运算
    function_type_3 = 'sum,min,max,mean'.split(',')  # 多元运算
    function_type_5 = 'Concat,Diff,Accumulate,Combine,Quantile_bin,Custom_Quantile_bin'.split(',')  # 自定义函数
    function_type_4 = '+,-,*,/'.split(',')  # 二元运算符
    commands = function_type_1 + function_type_2 + function_type_3  # + function_type_5
    keywords = function_type_1 + function_type_2 + function_type_3 + function_type_4
    keywords = set(keywords)
    commands = set(commands)

    token_specification = [
        ('NUMBER', r'\d+(\.\d*)?'),
        ('ASSIGN', r'='),
        ('END', r'$|;|#'),
        ('ID', r'@[A-Za-z0-9_:\u4E00-\u9FA5]+'),
        ('FUNC', r'[A-Za-z]+[0-9]*_?[A-Za-z0-9_]*'),
        ('OP', r'[+\-*/]'),
        ('SKIP', r'[ \t]+'),
        ('SPLIT', r'[\,]'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LC', r'\['),
        ('RC', r'\]'),
        ('MISMATCH', r'.'),
    ]
    env = dict()
    env['pi'] = np.pi
    env['e'] = np.e

    def __init__(self, df: pd.DataFrame):
        self.new_variables = dict(zip([f'_{f}' for f in df.columns], map(lambda field: df[field], df.columns)))
        self.env.update(self.new_variables)
        self.result_variables = dict()

        self.env['min'] = self._py_min
        self.env['max'] = self._py_max
        self.env['sum'] = self._py_sum
        self.env['mean'] = self._py_mean

        self.env['eq'] = self._py_eq
        self.env['exp'] = self._py_exp

        self.env['log2'] = self._py_log2
        self.env['log10'] = self._py_log10
        self.env['ln'] = self._py_ln
        self.env['log'] = self._py_log
        self.env['pow'] = self._py_pow

        # Concat,Diff,Accumulate,Combine,Quantile_bin,Custom_Quantile_bin
        self.env['Concat'] = self._py_concat
        self.env['Diff'] = self._py_diff
        self.env['Accumulate'] = self._py_acc
        self.env['Combine'] = self._py_combine
        self.env['Quantile_bin'] = self._py_quantile_bin
        self.env['Custom_Quantile_bin'] = self._py_custom_quantile_bin

    @de_func
    def _py_custom_quantile_bin(self, variable_list: list, *ops):
        rs = []
        # op 1 frequency 2 value
        for op in ops:

            for v in variable_list:
                min_value = v.min()
                max_value = v.max()
                op = np.clip(op, min_value, max_value)
                op = sorted(set(op))
                if op[0] > min_value:
                    op.insert(0, min_value)
                if op[-1] < max_value:
                    op.insert(len(op), max_value)
                if len(op) < 2:
                    op = [min_value, max_value]
                rs.append(pd.cut(v, op, right=True, include_lowest=True))

        return rs

    @de_func
    def _py_quantile_bin(self, variable_list: list, *ops):
        rs = []
        # op 1 frequency 2 value
        bins = ops[0]
        if bins <= 1:
            raise Exception('bins master bigger than 1')

        options = ops[1:]
        for op in options:
            for v in variable_list:
                if op == 1:
                    rs.append(pd.qcut(v, bins, duplicates='drop'))
                elif op == 0:
                    rs.append(pd.cut(v, bins, right=True, include_lowest=True))
                else:
                    raise Exception('quantile_bin parameter error')
        return rs

    @de_func
    def _py_combine(self, variable_list: list, *ops):
        rs = []
        for op in ops:
            rs.append(op(variable_list))
        return rs

    @de_func
    def _py_acc(self, variable_list: list, *ops):
        assert (len(ops) == 0)
        rs = []
        for v in variable_list:
            rs.append(v.cumsum())
        return rs

    @de_func
    def _py_diff(self, variable_list: list, *ops):
        rs = []
        for op in ops:
            assert (op >= 1)
            for v in variable_list:
                rs.append(v.diff(op))
        return rs

    @de_func
    def _py_concat(self, variable_list: list, *ops):
        rs = []
        for op in ops:
            assert (op > 1)
            for cs in combinations(range(len(variable_list)), op):
                vs = [variable_list[c] for c in cs]
                r = pd.concat(vs, axis=1).apply(lambda x: '_'.join(np.array(x).astype("str")), axis=1)
                rs.append(r)
        return rs

    @de_func
    def _py_log(self, variable_list: list, *ops):
        assert (len(ops) == 1)
        if isinstance(variable_list, BaseType):
            return np.log(variable_list) / np.log(ops[0])
        else:
            assert (len(variable_list) == 1)
            return np.log(variable_list[0]) / np.log(ops[0])

    @de_func
    def _py_pow(self, variable_list: list, *ops):
        assert (len(ops) == 1)
        if isinstance(variable_list, BaseType):
            return np.power(variable_list, ops[0])
        else:
            assert (len(variable_list) == 1)
            return np.power(variable_list[0], ops[0])

    @de_func
    def _py_log2(self, variable_list: list, *ops):
        assert (len(ops) == 0)
        if isinstance(variable_list, BaseType):
            return np.log2(variable_list)
        else:
            assert (len(variable_list) == 1)
            return np.log2(variable_list[0])

    @de_func
    def _py_log10(self, variable_list: list, *ops):
        assert (len(ops) == 0)
        if isinstance(variable_list, BaseType):
            return np.log10(variable_list)
        else:
            assert (len(variable_list) == 1)
            return np.log10(variable_list[0])

    @de_func
    def _py_ln(self, variable_list: list, *ops):
        assert (len(ops) == 0)
        if isinstance(variable_list, BaseType):
            return np.log(variable_list)
        else:
            assert (len(variable_list) == 1)
            return np.log(variable_list[0])

    @de_func
    def _py_eq(self, variable_list: list, *ops):
        if isinstance(variable_list, BaseType):
            return variable_list == ops[0]
        elif len(variable_list) == 2:
            return variable_list[0] == variable_list[1]
        elif len(variable_list) == 1 and len(ops) == 1:
            return variable_list[0] == ops[0]
        else:
            raise Exception('parameter error')

    @de_func
    def _py_exp(self, variable_list: list, *ops):
        assert (len(ops) == 0)
        if isinstance(variable_list, BaseType):
            return np.exp(variable_list)
        assert (len(variable_list) == 1)
        return np.exp(variable_list[0])

    @de_func
    def _py_min(self, variable_list: list, *ops):
        if isinstance(variable_list, BaseType):
            if pd.Series not in [type(op) for op in ops]:
                v = [variable_list]
                for op in ops:
                    v.append(op)
                return np.min(v)
            else:
                variable_list = [pd.Series([variable_list] * p_rows)]
        else:
            variable_list = [i for i in variable_list]

        for op in ops:
            if isinstance(op, BaseType):
                variable_list.append(pd.Series([op] * p_rows))
            else:
                variable_list.append(op)

        if len(variable_list) == 1:
            return np.min(variable_list[0])
        else:
            return np.min(pd.concat(variable_list, axis=1), axis=1)

    @de_func
    def _py_max(self, variable_list: list, *ops):
        if isinstance(variable_list, BaseType):
            if pd.Series not in [type(op) for op in ops]:
                v = [variable_list]
                for op in ops:
                    v.append(op)
                return np.max(v)
            else:
                variable_list = [pd.Series([variable_list] * p_rows)]
        else:
            variable_list = [i for i in variable_list]

        for op in ops:
            if isinstance(op, BaseType):
                variable_list.append(pd.Series([op] * p_rows))
            else:
                variable_list.append(op)

        if len(variable_list) == 1:
            return np.max(variable_list[0])
        else:
            return np.max(pd.concat(variable_list, axis=1), axis=1)

    @de_func
    def _py_mean(self, variable_list: list, *ops):
        if len(variable_list) == 1:
            return np.mean(variable_list[0])
        else:
            return np.mean(pd.concat(variable_list, axis=1), axis=1)

    @de_func
    def _py_sum(self, variable_list: list, *ops):
        if len(variable_list) == 1:
            return np.sum(variable_list[0])
        else:
            return np.sum(pd.concat(variable_list, axis=1), axis=1)

    def tokenize(self, code):
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in self.token_specification)
        line_num = 1
        line_start = 0
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
            elif kind == 'SKIP':
                pass
            elif kind == 'MISMATCH':
                raise RuntimeError(f'{value!r} unexpected on line {line_num}')
            else:
                column = mo.start() - line_start
                if kind == 'ID':
                    value = value[1:]
                if kind == "END":
                    value = "END"
                yield Token(kind, value, line_num, column)

    def checker(self, serial):
        """
        语法检查
        :param serial: 
        :return: 
        """
        ss = []
        for s in serial:
            if s.type == 'END':
                break

            ss.append(s)

        if len(ss) < 3 or len(ss) == 0:
            return []

        index_assign = -1
        for i in range(len(ss)):
            if ss[i].type == 'ASSIGN':
                index_assign = i

        if index_assign < 0:
            raise Exception(f"should assign a variable")

        # 赋值检查
        result_ids = [s for s in ss[:index_assign] if s.type == 'ID']
        if len(result_ids) == 0:
            raise Exception(f"should assign a variable")

        # 常规语法检查
        # for i in range(len(ss) - 1):
        #     if ss[i].type == ss[i + 1].type:
        #         raise Exception(f'syntax error at index {ss[i + 1].column}')

        # 语法1
        ss_3 = ss[:index_assign + 1]
        for s in ss[index_assign + 1:]:
            if s.type == 'ID':
                if f'_{s.value}' not in self.new_variables:
                    raise Exception(f'un define variable {s.value}')
                ss_3.append(s)
            else:
                ss_3.append(s)

        # 常规语法检查2
        for i in range(len(ss_3) - 1):
            if ss_3[i + 1].value == '(' and ss_3[i].type not in {'COMMAND', 'FUNC', 'ASSIGN'}:
                raise Exception('syntax error')

        # 括号
        brackets = []
        for s in ss_3[2:]:
            if s.type == 'LPAREN':
                brackets.insert(0, 'lc')
            elif s.type == 'RPAREN':
                brackets.pop(0)
        if len(brackets) > 0:
            raise Exception('syntax error in "(" or ")"')

        brackets = []
        for s in ss_3[2:]:
            if s.type == 'LC':
                brackets.insert(0, 'lc')
            elif s.type == 'RC':
                brackets.pop(0)

        if len(brackets) > 0:
            raise Exception('syntax error in "[" or "]"')

        # 安全限制
        if len(ss_3) > 40:
            raise Exception("your expression is to long")

        return ss_3

    def execute(self, serial, line):
        index_assign = -1
        for i in range(len(serial)):
            if serial[i].type == 'ASSIGN':
                index_assign = i

        result_variable = [str(s.value) for s in serial[:index_assign] if s.type == 'ID']

        eval_str = ''.join([s.value if s.type != 'ID' else f'_{s.value}' for s in serial[index_assign + 1:]])

        try:
            rs = eval(eval_str, self.env)  # gai hou dataframe
        except Exception as e:
            raise Exception(f'syntax error\n{e}')

        if not isinstance(rs, list):
            if isinstance(rs, pd.DataFrame):
                rs = [rs.iloc[:, i] for i in range(len(rs.columns))]
            elif isinstance(rs, BaseType):
                rs = [pd.Series([rs] * p_rows)]
            else:
                rs = [rs]

        if len(rs) != len(result_variable):
            raise Exception(f'can not assign {len(result_variable)} result with {len(rs)} return data')

        for r in rs:
            # if not isinstance(r, float) and not isinstance(r, np.ndarray) and not isinstance(r, pd.Series):
            if not isinstance(r, np.ndarray) and not isinstance(r, pd.Series):
                raise Exception('syntax error')

        for name, value in zip(result_variable, rs):
            self.env[name] = value
            if isinstance(value, np.ndarray):
                value = pd.DataFrame(value, columns=[name])
            if isinstance(value, (pd.Series, pd.DataFrame)):
                try:
                    self.result_variables[name] = self.new_variables[name] = pd.DataFrame(value.values, columns=[name])
                except:
                    self.result_variables[name] = self.new_variables[name] = pd.DataFrame(list(value.values),
                                                                                          columns=[name])
                    # self.new_variables[name] = pd.DataFrame(value, columns=[name])

    def _run(self, scripts):

        ss = [s.strip() for s in scripts.split(';')]

        for line, s in enumerate(ss):
            if s == '' or s[0] in {'#', ';'}:
                continue

            try:
                if hasattr(self, f'_serial_{line}'):
                    serial = getattr(self, f'_serial_{line}')
                else:
                    serial = self.tokenize(s)
                    serial = self.checker(serial)
                    setattr(self, f'_serial_{line}', serial)

                if not serial:
                    continue
                self.execute(serial, line + 1)
            except Exception as e:
                traceback.print_exc()
                raise Exception(f'line {line + 1} tokenize error:{e} \n {s}')

    def _decode(self, serial):
        """
        解析原始变量，新变量
        :param serial:
        :return:
        """
        index_assign = -1
        for i in range(len(serial)):
            if serial[i].type == 'ASSIGN':
                index_assign = i

        result_variable = [s.value for s in serial[:index_assign] if s.type == 'ID']
        base_variable = [s.value for s in serial[index_assign:] if s.type == 'ID']
        return base_variable, result_variable

    def decode(self, scripts):

        ss = [s.strip() for s in scripts.split(';')]
        b, n = [], []
        for line, s in enumerate(ss):
            if s == '' or s[0] in {'#', ';'}:
                continue

            try:
                if hasattr(self, f'_serial_{line}'):
                    serial = getattr(self, f'_serial_{line}')
                else:
                    serial = self.tokenize(s)
                    serial = self.checker(serial)
                    setattr(self, f'_serial_{line}', serial)

                if not serial:
                    continue
                base_variable, result_variable = self._decode(serial)
                b.extend(base_variable)
                n.extend(result_variable)

            except Exception as e:
                traceback.print_exc()
                raise Exception(f'line {line + 1} tokenize error:{e} \n {s}')
        return b, n

    def run(self, scripts):
        try:
            self._run(scripts)
        except Exception as e:
            raise e


if __name__ == '__main__':
    import pytest

    # df = pd.DataFrame(np.eye(3, 3), columns=['var1', 'var2', 'var3'])

    df = pd.DataFrame(np.random.randn(100, 6) * 100,
                      columns=['var', '123', 'var_123', 'var:123', 'var+123', 'var 123'])

    s1 = '@n1=Accumulate(@var)+1'
    s2 = '@n1=Concat(Accumulate(@var,@var_123),2)'
    s3 = '@n1=@var+@123+@var_123'
    s4 = '@n1=min(exp(@var+1),@123)'
    s5 = '@n1=log(@var,10)'
    s6 = '@n1=ln(exp(ln(exp(@var))))'
    s7 = '@n1,@n2,@n3=Concat(Concat(@var,@123,@var_123,2),2)'
    s8 = '@n1=max(0,@var,100,@123)'
    s9 = '@n1,@n2=Diff(@var,1,2)'
    s10 = '@n1,@n2,@n3,@n4=Quantile_bin(@var,@var_123,3,1,0)'
    s11 = '@n1=min(5,1*10,1)/5*exp(1)/ln(e)/e'
    s12 = '@n1=Custom_Quantile_bin (@var,[0,50])'
    
    p_rows=len(df)
    ts = TransformScripts(df)
    ts.run(s12)
    print(ts.result_variables)


    def S1():
        ts = TransformScripts(df)
        ts.run(s1)
        assert all(df['var'].cumsum() + 1 == ts.result_variables['n1']['n1'])


    def S2():
        ts = TransformScripts(df)
        ts.run(s2)
        v = pd.concat([df['var'], df['var_123']], axis=1).cumsum()
        vs = [v.iloc[:, c] for c in (0, 1)]
        r = pd.concat(vs, axis=1).apply(lambda x: '_'.join(np.array(x).astype("str")), axis=1)
        assert all(r == ts.result_variables['n1']['n1'])


    def S3():
        ts = TransformScripts(df)
        ts.run(s3)
        assert all(df['var'] + df['123'] + df['var_123'] == ts.result_variables['n1']['n1'])


    def S4():
        ts = TransformScripts(df)
        ts.run(s4)
        r = np.min(pd.concat([np.exp(df['var'] + 1), df['123']], axis=1), axis=1)
        assert all(r == ts.result_variables['n1']['n1'])


    def S5():
        ts = TransformScripts(df)
        ts.run(s5)
        r = (np.log(df['var']) / np.log(10)).fillna(0)
        ts = ts.result_variables['n1']['n1'].fillna(0)
        assert all(r == ts)

    def S6():
        ts = TransformScripts(df)
        ts.run(s6)
        r=[round(i,2) for i in df['var']]
        ts=[round(i,2) for i in ts.result_variables['n1']['n1']]
        assert r==ts
        
    def S7():
        ts = TransformScripts(df)
        ts.run(s7)        
        v = pd.concat([df['var'],df['123'],df['var_123']], axis=1)
        vs1 = [v.iloc[:, c] for c in (0, 1)]
        vs2 = [v.iloc[:, c] for c in (0, 2)]
        r1=pd.concat(vs1, axis=1).apply(lambda x: '_'.join(np.array(x).astype("str")), axis=1)
        r2=pd.concat(vs2, axis=1).apply(lambda x: '_'.join(np.array(x).astype("str")), axis=1)
        r=pd.concat([r1,r2], axis=1).apply(lambda x: '_'.join(np.array(x).astype("str")), axis=1)
        assert all(r == ts.result_variables['n1']['n1'])

    def S8():
        ts = TransformScripts(df)
        ts.run(s8)
        r=np.max(pd.concat([pd.Series([0]*100),df['var'],pd.Series([100]*100),df['123']],axis=1),axis=1)
        assert all(r == ts.result_variables['n1']['n1'])       
 
    def S9():
        ts = TransformScripts(df)
        ts.run(s9)
        r=df['var'].diff(2)
        assert all(r.fillna(0) == ts.result_variables['n2']['n2'].fillna(0)) 

    def S10():
        ts = TransformScripts(df)
        ts.run(s10)
        r=pd.cut(df['var'],3, right=True, include_lowest=True)
        assert all(r == ts.result_variables['n3']['n3']) 
        
    def S11():
        ts = TransformScripts(df)
        ts.run(s11)
        r=pd.Series([0.2]*100)
        assert all(r == ts.result_variables['n1']['n1'])
        
    def S12():
        ts = TransformScripts(df)
        ts.run(s12)
        v=df['var']
        min_value = v.min()
        max_value = v.max()
        op=np.clip([0,50], min_value, max_value)
        op = sorted(set(op))
        if op[0] > min_value:
            op.insert(0, min_value)
        if op[-1] < max_value:
            op.insert(len(op), max_value)
        if len(op) < 2:
            op = [min_value, max_value]
        r=pd.cut(v, op, right=True, include_lowest=True)
        assert all(r == ts.result_variables['n1']['n1'])        
        
    Test_S = [S1, S2, S3, S4, S5,S6,S7,S8,S9,S10,S12]
    for S in Test_S:
        S()

