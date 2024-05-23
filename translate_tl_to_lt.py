

fn="answer-yulun-sub.txt"
gn=f"{fn.split('.')[0]}-left_top_comma.txt"
with open(fn,'r') as f, \
        open(gn,'w') as g:
    for line in f:
        frame,bid,top,left,width,height,_,_,_,_=line.split()
        g.write(f"{frame} {bid} {left} {top} {width} {height} -1 -1 -1 -1\n")
        # g.write(f"{frame}, {bid}, {left}, {top}, {width}, {height}, -1, -1, -1, -1\n")
