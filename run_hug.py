import hug
import os
import errno
from read_credit_card_hug import *
import cv2
import imageio
import uuid
import random
import re
from PIL import Image
import io
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



@hug.static('/images_generated')
def static_output():
    return ('./app/static/output', )


@hug.post('/credit_card',output=hug.output_format.html)
def creditCard(front_image, back_image,billing_zip, cors: hug.directives.cors="*"):
    print("Resizing and saving images...")

    billing_zip = billing_zip.decode('UTF-8') 

    img_f = Image.open(io.BytesIO(front_image))
    img_b = Image.open(io.BytesIO(back_image))

    im_width = 416

    #Unique identifier
    u_id_ = str(uuid.uuid4()) 
    unique_id = u_id_ + ".jpg"
    
    # resize and save image
    
    image_f_resized = 'app/static/output/f_image_'+ unique_id
    image_b_resized = 'app/static/output/b_image_'+ unique_id

    json_file = 'app/static/output/credit_card_results.json'



    
    # Image size
    width, height = img_f.size
    fra = width/height
    im_height = int(im_width/fra)

    img_f  = img_f.resize((im_width, im_height), Image.ANTIALIAS)
    img_f  = img_f.convert("RGB")
    img_f.save(image_f_resized)

        # Image size
    width, height = img_b.size
    fra = width/height
    im_height = int(im_width/fra)

    img_b  = img_b.resize((im_width, im_height), Image.ANTIALIAS)
    img_b  = img_b.convert("RGB")
    img_b.save(image_b_resized)

    res_image_path, results_dict = read_credit_card(image_f_resized,unique_id,"f")
    print("##################")
    print("results_dict:", results_dict )
    print("##################")

    found = True
    found_name = True
    found_num = True
    found_ex = True

    if 'expire' not in results_dict:
        found = False
        found_ex = False

    if 'name' not in results_dict:
        found = False
        found_name = False


    if 'number' not in results_dict:
        found = False
        found_num = False


    if found==False:
        res_image_path2, results_dict2 = read_credit_card(image_b_resized,unique_id,"b")

        print("#########Back of the image:")
        print("results_dict2:",results_dict2)


    if found:
        out_text = "All Information Extracted!"
    else:
        out_text = "All Information Cannot be Found! Please Upload Images Again!"

    res_image_path_app = re.sub('app/static/output','http://localhost:8000/images_generated', res_image_path.rstrip())


    results_dict['unique_id'] = u_id_
    results_dict['front_image'] = image_f_resized
    results_dict['back_image'] = image_b_resized
    results_dict['processed_image'] = res_image_path
    results_dict['processed_image_app'] = res_image_path_app
    results_dict['billing_zip'] = billing_zip

    if os.path.exists(json_file):  
        with open(json_file) as f:
            data = json.load(f)

        data.update(results_dict)
    else:
        data = results_dict

    with open(json_file, 'w') as outfile:  
        json.dump(data, outfile)


    json_results = json.dumps(results_dict, indent=2)

    html = """<HTML>
                <body>
                    <h1>Credit Card Read</h1>

                    <h2>m{message}</h2>
                    <h3>Results</h3>  
                    <b>Extracted</b><br>
                    <pre>{j_results}</pre>

                    <h3>Result Images</h3> 
                    <h5>Front</h5>
                    <img src="/images_generated/f_image_{u_id}" style="height:300px;">
                    <img src="/images_generated/processed_f{u_id}" style="height:300px;">
                    <h5>Back</h5>
                    <img src="/images_generated/b_image_{u_id}" style="height:300px;">
                    <img src="/images_generated/processed_b{u_id}" style="height:300px;">


                </body>
                </HTML>"""    

    #return json_results   
    return html.format(u_id = unique_id,j_results=json_results, message = out_text)



@hug.get('/input_data', output=hug.output_format.html)
def input_form(cors: hug.directives.cors="*"):
    # https://www.curaition.io/api/photo/play
    return ('<form action="/credit_card" enctype="multipart/form-data" method="post">'+
    'Front of the credit card: <input type="file" name="front_image" multiple="multiple"><br>'+
    'Back of the credit card: <input type="file" name="back_image" multiple="multiple"><br>'+
    'Zip Code: <input name="billing_zip"><br><br>'+
    '<input type="submit" value="Upload"></form>')






