from marshmallow import Schema, fields

class ResponseMessageSchema(Schema):
    message = fields.Str()