from sentence_splitter_for_russian import Ru_splitter

splitter = Ru_splitter.Splitter()

testfile = open("testtext.txt", 'r')
text = testfile.read().strip()
testfile.close()

res = splitter.split(text)
for i in res:
    print(i)
