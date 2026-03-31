from flask import Flask, request, session, make_response, render_template

app = Flask(__name__)
app.secret_key = '1234'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['usuario'] = request.form['nome']
        response = make_response(render_template(
            'index.html', usuario=session['usuario']))
        response.set_cookie('usuario', session['usuario'])
        return response
    else:
        usuario = session.get('usuario', 'visitante')
        cookie_usuario = request.cookies.get('usuario', 'sem cookie')
        return render_template('index.html', usuario=usuario, cookie_usuario=cookie_usuario)


if __name__ == '__main__':
    app.run(debug=True)
