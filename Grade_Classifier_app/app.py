import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, request
#pd.options.display.max_columns=25

app = Flask(__name__, static_url_path='/static')

#Standard home page. 'index.html' is the file in your templates that has the CSS and HTML for your app
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

#My home page redirects to recommender.html where the user fills out a survey (user input)
@app.route('/input_holds', methods=['GET', 'POST'])
def recommender():
    return render_template('input_holds.html')

#After they submit the survey, the recommender page redirects to recommendations.html
@app.route('/return_grade', methods=['GET', 'POST'])
def input_to_output_app():
	import pickle
	filename = 'RF_imbmodel.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	import seaborn as sns
	font = ['5+','6A','6A+','6B','6B+','6C','6C+','7A','7A+','7B','7B+','7C','7C+','8A','8A+','8B','8B+']
	vgrd = [2,3,3,4,4,5,5,6,7,8,8,9,10,11,12,13,14]
	grade_conversion = dict(zip(font,vgrd))
	def alphabet_to_num(char):
		return ord(char.lower()) - 96
	def split_xy(string):
		import re
		r = re.compile("([a-zA-Z]+)([0-9]+)")
    #strings = ['foofo21', 'bar432', 'foobar12345']
		return list(r.match(string).groups())
	def xlims(list):
    # return (leftmost, rightmost) x coordinate of a given problem
		n = len(list)
		coords = [alphabet_to_num(split_xy(list[i])[0]) for i in range(n)]
#    xcoords = [coords[i][0] for i in range(n)]
#    xcoords = [alphabet_to_num(coords[i]) for i in range(n)]
		return [min(coords),max(coords)]
# generate classes for mlb!
	mlb_class = []
	for i in range(11):
		for j in range(18):
			mlb_class.append(chr(i+65)+'%d'%(j+1))
	from sklearn.preprocessing import MultiLabelBinarizer

	mlb = MultiLabelBinarizer(classes=mlb_class)

# Using Skicit-learn to split data into training and testing sets
# Instantiate model with 1000 decision trees
# Train the model on training data

	def input_to_output(list):
		h_enc = mlb.fit_transform([list])
		h_enc = pd.DataFrame(h_enc,columns=mlb.classes_)
		tmp = xlims(list)
		tmp = {'lr':[tmp]}
		tmp2 = pd.DataFrame.from_dict(tmp)
		tmp2[['l','r']] = pd.DataFrame(tmp2.lr.values.tolist(), index= tmp2.index)
		tmp2 = tmp2.drop(['lr'],axis=1)
		h_enc['Length'] = pd.Series(len(list))
		h_enc = h_enc.join(tmp2)
		h_enc['width'] = h_enc['r']-h_enc['l']
		prob = loaded_model.predict(h_enc)
		#return [rf.predict(h_enc),rf.predict_proba(h_enc)]
		return prob

    # take input from recommendation.html
	s = str(request.form['hold_coordinates'])
	lst = s.strip().split(',')
	cos_sim = input_to_output(lst)
    # plot can be generated, saved as a file and loaded to html
    #return [rf2.predict(h_enc),rf2.predict_proba(h_enc)]
    #return render_template('recommendations.html', cos_sims = cos_sims, florist_info = florist_info)
    # return the calculation to recommendations.html
	return render_template('return_grade.html', cos_sim = cos_sim)#,  florist_info = b)

if __name__ == '__main__':
    #this runs your app locally
    app.run(host='127.0.0.1', port=8080, debug=True)
    #from werkzeug.serving import run_simple
    #run_simple('localhost', 8080, app)
