a,b,c,d=1,1,1,1
count=0
while a<=10:
    while b<=10:
            count+=10
            b+=2
            while c<=10:
                count+=20
                c+=4
                while d<=10:
                    count+=30
                    d+=3
            c=1
    b,c,d=1,1,1
    a+=2
    count+=40
print(a,b,c,d,count)