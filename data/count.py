with open('data.txt', 'r', encoding='utf-8') as f:
    leng = 0
    for line in f.readlines():
        leng += len(line.strip().split())
    print(leng)
