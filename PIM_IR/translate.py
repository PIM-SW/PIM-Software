# readline_all.py
f = open("./out.mlir", 'r')
import random
base=random.random()
offset=0
base=int(base*100)*100
last=""
lines = f.readlines()
inputs=[]
for line in lines:    
    if not line: break
    line=line.lstrip()
    if line:
        if line[0]=='%':
            cur_line=line.split(')')[0]
            inputs=inputs+cur_line.split('(')[-1].split()
in_args=set(inputs)
load_list=[]
for arg in in_args:
    if arg[0]=='%':
        load_list.append(arg)

for line in lines:    
    if not line: break
    line=line.lstrip()
    if line:
        if '@' in line:
            cur_line=line.split('@')[1]
            cur_line=cur_line.split(')')[0]
            cur_line=cur_line.split('(')[-1]
            objs=cur_line.split(',')
            for obj in objs:
                objname, objty =obj.split(':')
                objname=objname.lstrip()
                objty=objty.lstrip()
                if(objname in load_list):
                    print('LOAD '+objname+' #'+str(base+offset))
                if('<' in objty):
                    objty=objty.split('>')[0]
                    objty=objty.split('<')[1]
                objty=objty.replace('i32', '4')
                objty=objty.replace('i64', '8')
                objty=objty.replace('f32', '4')
                objty=objty.replace('f64', '8')
                if('x' in objty):
                    res=1
                    objty=objty.split('x')
                    for cs in objty:
                        res*=int(cs)
                    objty=str(res)
                offset+=int(objty)
        if line[0]=='%':
            cur_line=line.split(')')[0]
            res, cur_line=cur_line.split('=')
            op=line.split('(')[0]
            op=op.split('.')[-1].upper()
            ins=cur_line.split('(')[-1]
            last=res
            print(op+" "+res+ins)
print("STORE "+last+" #"+str(base+offset))
f.close()

