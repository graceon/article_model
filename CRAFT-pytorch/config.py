CLASS=[
'0',
'1',
'2',
'3',
'4',
'5',
'6',
'7',
'8',
'9',
'add',
'sub',
'mul',
'div',
'equal',
]
def CLASS_is_op(pred):
    return pred>=10 and pred<=14
def CLASS_is_eq(pred):
    return pred==14

CLASS_toString=[
'0',
'1',
'2',
'3',
'4',
'5',
'6',
'7',
'8',
'9',
'+',
'-',
'x',
'/',
'='
]

CLASS_base=[
    'print',
    'writing'
]

N_CLASS=len(CLASS)

NET_DEPTH=16