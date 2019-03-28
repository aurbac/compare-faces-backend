import json
import urllib
import boto3
import os
import base64

import PIL.ImageDraw
import PIL.ImageFont
import PIL.Image

def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def DrawBox(myimage,draw,boundingBox,width,linecolor,name):
    boxLeft = float(boundingBox['Left'])
    boxTop = float(boundingBox['Top'])
    boxWidth = float(boundingBox['Width'])
    boxHeight = float(boundingBox['Height'])
    imageWidth = myimage.size[0]
    imageHeight = myimage.size[1]
    rectX1 = imageWidth * boxLeft
    rectX2 = imageHeight * boxTop
    rectY1 = rectX1 + (imageWidth * boxWidth)
    rectY2 = rectX2 + (imageHeight * boxHeight)
    fontSans = PIL.ImageFont.truetype("FreeSans.ttf", 20)
    draw.text((rectX1 + width,rectX2 + width), name, fill=(255,252,102),font=fontSans)
    for i in range(0,width):
        draw.rectangle(((rectX1 + i,rectX2 + i),(rectY1 - i,rectY2 - i)), fill=None, outline=linecolor)
        
def DrawBoxText(myimage,draw,boundingBox,width,linecolor):
    boxLeft = float(boundingBox['Left'])
    boxTop = float(boundingBox['Top'])
    boxWidth = float(boundingBox['Width'])
    boxHeight = float(boundingBox['Height'])
    imageWidth = myimage.size[0]
    imageHeight = myimage.size[1]
    rectX1 = imageWidth * boxLeft
    rectX2 = imageHeight * boxTop
    rectY1 = rectX1 + (imageWidth * boxWidth)
    rectY2 = rectX2 + (imageHeight * boxHeight)
    for i in range(0,width):
        drawrect(draw, [(rectX1 + i,rectX2 + i), (rectY1 - i,rectY2 - i)], outline=linecolor, width=0)


def getInformation(event, context):
    print(json.dumps(event))
    params = base64.b64decode(event['body'])
    data_keys = json.loads(params)
    
    bucket = os.environ['S3_BUCKET']
    
    rk = boto3.client('rekognition','us-east-1')
    
    source_key = data_keys['key_image_source']
    target_key = data_keys['key_image_target']
    download_path = '/tmp/input.jpg'
    upload_path = '/tmp/output.jpg'
    
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, target_key, download_path)
    
    text_detections = []
    
    response_detect_faces_source = rk.detect_faces(
        Image={
            'S3Object': {
                'Bucket': bucket,
                'Name': source_key
            }
        },
        Attributes=['DEFAULT']
    )
    
    print(json.dumps(response_detect_faces_source))
    
    response_detect_faces_target = rk.detect_faces(
        Image={
            'S3Object': {
                'Bucket': bucket,
                'Name': target_key
            }
        },
        Attributes=['DEFAULT']
    )
    
    print(json.dumps(response_detect_faces_target))
    
    
    if len(response_detect_faces_source['FaceDetails'])>0 and len(response_detect_faces_target['FaceDetails'])>0:
        
        response_text = rk.detect_text(
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': target_key
                }
            }
        )
        
        print(json.dumps(response_text))
        
        response = rk.compare_faces(
            SourceImage={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': source_key
                }
            },
            TargetImage={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': target_key
                }
            },
            SimilarityThreshold=1
        )
        
        print(json.dumps(response))
        
        faceMatches = response['FaceMatches']
        with PIL.Image.open(download_path) as myimage:
            draw = PIL.ImageDraw.Draw(myimage)
            for text in response_text['TextDetections']:
                if text['Confidence']>70 and text['Type']=='LINE':
                    text_detections.append(text['DetectedText'])
                    DrawBoxText(myimage,draw,text['Geometry']['BoundingBox'],4,(0,112,255))
            
            for celeFace in faceMatches:
                Face = celeFace['Face']
                name = str(float("{0:.2f}".format(celeFace['Similarity'])))+'%'
                print(name)
                boundingBox = Face['BoundingBox']
                print(boundingBox)
                DrawBox(myimage,draw,boundingBox,4,(36,157,61),name)
            myimage.save(upload_path)
        
        s3_client.upload_file(upload_path,bucket, 'output/'+data_keys['id']+'-result.jpg')
        
        print(json.dumps(text_detections))
        print(json.dumps(event))
    
        body = {
            "text_detections": text_detections,
            "face_in_source" : True,
            "face_in_target" : True
        }
    
        response = {
            "statusCode": 200,
            "headers": {
                    'Access-Control-Allow-Origin': '*'
            },
            "body": json.dumps(body)
        }
        
        return response
    else:
        face_in_source = False
        face_in_target = False
        if len(response_detect_faces_source['FaceDetails'])>0:
            face_in_source = True
        if len(response_detect_faces_target['FaceDetails'])>0:
            face_in_target = True
        
        body = {
            "text_detections": None,
            "face_in_source" : face_in_source,
            "face_in_target" : face_in_target,
        }
    
        response = {
            "statusCode": 200,
            "headers": {
                    'Access-Control-Allow-Origin': '*'
            },
            "body": json.dumps(body)
        }
        
        return response



def getImageResult(event, context):
    print(json.dumps(event))
    bucket = os.environ['S3_BUCKET']
    
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }
    
    print(json.dumps(event))
    
    image_key = 'output/'+event['pathParameters']['id']+'-result.jpg'
    image_path = '/tmp/image.jpg'
    
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, image_key, image_path)
    
    response = {
        "isBase64Encoded": True,
        "statusCode": 200,
        "headers": {
                'Content-Type': 'image/jpeg', 'Accept' : 'image/jpeg'
        },
        "body": base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
    }

    return response

