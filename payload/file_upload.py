from marshmallow import Schema, fields

class FileUploadSchema(Schema):
    file = fields.Raw(type='file', required=True)