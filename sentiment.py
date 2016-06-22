import sentiment_analysis as s

text = raw_input("Enter your tweet...\n")

string = str(text)
answer = s.final(string)

if answer[0] == 1:
	x = 'Positive'
else:
	x = 'Negative'

print "Your tweet is:", x, ", with confidence value:", round(answer[1],2), "%"