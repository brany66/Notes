#!/usr/bin/env python                                                                                                                                
# encoding: utf-8
 
import sys 
 
reload(sys)
sys.setdefaultencoding("utf-8")
 
def full2half(ustring):
    """将全角转换成半角
    Parameter:
        -ustring:unicode
    return:
        -rstring:unicode
    """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:      #转完之后不是半角字符则返回原来的字符
            rstring += uchar #;  print '#%s-%s*' % (inside_code,uchar)
        else:
            rstring += unichr(inside_code)
    return rstring
 
for line in sys.stdin:
    line = line.strip().decode("utf-8")
    print full2half(line)

