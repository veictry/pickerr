import re


def hash_list(array, next=4, order_relate=True):
    varray = []
    for elem in array:
        if elem: varray.append(elem)
    l = len(varray)
    return sum([sum([elem * (i + j + 1) for j, elem in enumerate(varray[i:min(l, i + next)])])
                for i in range(l)]) \
        if order_relate else \
        sum([elem for elem in varray])

def hash_value(content, line_order_relate=False):
    return hash_list(
                [hash_list(
                    [hash_list([ord(c) for c in word])
                     for word in re.split(r'\s+', line)]
                 ) for line in re.split(r'\r?\n', content)]
            , order_relate=line_order_relate)

def dict_word(words):
    return dict([(v, i) for i, v in enumerate(set(words))])

def dict_line(line):
    return dict_word(re.split(r'\s+',line))

def compare(s1, s2):
    dict_item_str = lambda d: ''.join(sorted(dict_line(d).keys()))
    if dict_item_str(s1) == dict_item_str(s2):
        return True

if __name__ == '__main__':
    a = '''
    f d
    b 
    ddfffdfa sad
    afdasffd faa
    
    '''
    print(compare('a b', 'a b'))