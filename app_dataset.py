import flask, os, sys,time
from flask import request
from flask import render_template

interface_path = os.path.dirname(__file__)
sys.path.insert(0, interface_path)  #将当前文件的父目录加入临时系统变量

print(__name__)
server = flask.Flask(__name__, static_folder='res')
server = flask.Flask(__name__, static_folder='web')



lables_dir="resjson/"

lables=os.listdir(lables_dir)
print(lables)






@server.route('/<int:id>', methods=['get'])
def index(id):
    if(id>=len(lables) or id < 0):
        return "do not exist"
    json_name=lables[id]
    with open(lables_dir+json_name) as json_file:
        json=json_file.read()
    picture_name=json_name.split(".")[0]+'.png'
    rect_picutre='web/'+picture_name
    return render_template('app_dataset.html', json=json,rect_picutre=rect_picutre,next=id+1,prev=id-1,json_name=json_name,picture_name=picture_name)

# @server.route('/<int:id>', methods=['get'])
# def view_anno(id):
#     if(id>=len(lables) or id < 0):
#         return "do not exist"
#     json_name=lables[id]
#     with open(lables_dir+json_name) as json_file:
#         json=json_file.read()
#     rect_picutre='web/'+json_name.split(".")[0]+'.png'
#     return render_template('app_dataset.html', json=json,rect_picutre=rect_picutre,next=id+1,prev=id-1,json_name=json_name)





@server.route('/write_anno',methods = ["GET","POST"])
def write_anno():
    line = request.form.get('data[line]')
    with open("anno.txt", encoding="utf-8",mode="a") as file:  
        file.write(line+"\n")

    return "OK"

server.config['DEBUG'] = True
server.run(port=8001)

