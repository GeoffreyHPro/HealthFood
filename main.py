from flask import Flask
from flask_smorest import Api, Blueprint 
from flask.views import MethodView
from marshmallow import Schema, fields
from config_swagger.config import Config
from payload.file_upload import *
from responses.message_response import *
from services.ml import test_image, main

app = Flask(__name__)
app.config.from_object(Config)

api = Api(app)

blp = Blueprint(
    'examples', 
    'examples', 
    url_prefix='', 
    description='Operations on examples'
)


@blp.route('/train_model')
class Example(MethodView):

    @blp.response(200, ResponseMessageSchema)
    def get(self):
        """Entrainer le model de classification"""
        main()
        return { "message" : "finished"}

class ExampleResponseSchema(Schema):
    id = fields.Int()
    name = fields.Str()

@blp.route('/upload')
class FileUpload(MethodView):

    @blp.arguments(FileUploadSchema, location='files')
    @app.errorhandler(400)
    @blp.response(200)
    def post(self, file_data):
        """Evaluer une image en fonction du model"""
        uploaded_file = file_data['file']
        # Save the file or process it here
        filename = uploaded_file.filename
        return test_image(uploaded_file)
        #uploaded_file.save(f"./uploads/{filename}")
        #return test_image(f"./uploads/{filename}")


# Enregistrement des blueprints
api.register_blueprint(blp)
    

if __name__ == '__main__':
    app.run(debug=True)