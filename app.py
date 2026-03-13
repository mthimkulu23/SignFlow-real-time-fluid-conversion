from flask import Flask, render_template, Response
from controllers.main_controller import main_bp

import os
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

app = Flask(__name__, template_folder='templates', static_folder='templates/static')

# Register blueprints
app.register_blueprint(main_bp)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
