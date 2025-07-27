import os
import glob
import shutil
import app.config as config
import cv2 as cv
from PIL import Image
from deepface import DeepFace
from app.utils import remove_representation, check_empty_db
from app.database import get_db_collection # Import your MongoDB collection

from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, HTTPException, File, UploadFile

import numpy as np

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get MongoDB collection
mongo_collection = get_db_collection()

@app.get("/")
def root():
    '''
    Greeting!!!
    '''
    # Check MongoDB connection instead of file system
    try:
        mongo_collection.find_one({}) # Try to access the collection
        return {
            "message": "Welcome to Face Recognition API. MongoDB connected."
        }
    except Exception as e:
        return {
            "message": f"Error when trying to connect to MongoDB: {e}"
        }

@app.get('/img-db-info')
def get_img_db_info(return_img_file:bool | None = True):
    '''
    Get database information, return all files in the database
    '''
    # Fetch info from MongoDB
    all_records = list(mongo_collection.find({}, {"_id": 0, "filename": 1}))
    numer_of_images = len(all_records)
    
    if return_img_file:
        return {
            "number_of_image": numer_of_images,
            "all_images_file": [record["filename"] for record in all_records],
        }
    else:
        return {
            "number_of_image": numer_of_images,
        }


@app.get('/show_img/{img_path}')
def show_img(img_path: str | None = None):
    '''
    Return image file from given image name

    Arguments:
        img_path(str): image file
        return_image_name(bool): Decide whether return only image file (img) or image file with extension (img.[jpg|jpeg])
    '''
    # Check if database is empty via MongoDB
    if mongo_collection.count_documents({}) == 0:
        return "No image found in the database"

    if img_path is None:
        return {
            "error": "Client should provide image file name"
        }

    # Assuming img_path directly refers to the filename stored in the DB_PATH
    # You might want to store the full path in MongoDB if images are not always in DB_PATH
    full_img_path = os.path.join(config.DB_PATH, img_path)
    if not os.path.exists(full_img_path):
         raise HTTPException(status_code=404, detail=f"Image {img_path} not found in file system.")

    return FileResponse(full_img_path)


@app.post('/register')
def face_register(
    img_file: UploadFile | None = File(None, description="Upload Image"),
    to_gray: bool | None = Query(
            default=True,
            description="Whether save image in gray scale or not"),
    img_save_name: str | None = Query(
        default=None,
        description="File's name to be save, file extension can be available or not",
    ),):
    '''
    Add new user to the database by face registering. Resize image if necessary.

     Arguments:
        img_file(File): upload image file
        img_save_name(string): name of image file need to be saved
    '''
    if img_file is None:
        return {
            "message": "Image file need to be sent!",
        }

    save_img_dir = ''
    filename_to_save = ''

    # Determine the filename to save
    if img_save_name:
        extension = img_file.filename.split(".")[-1]
        if "." in img_save_name:
            img_save_name_extension = img_save_name.split(".")[-1]
            if extension != img_save_name_extension:
                raise HTTPException(status_code=404, detail='File extension should match')
            filename_to_save = img_save_name
        else:
            filename_to_save = img_save_name + "." + extension
    else:
        if '/' in img_file.filename:
            filename_to_save = img_file.filename.split('/')[-1]
        elif "\\" in img_file.filename:
            filename_to_save = img_file.filename.split("\\")[-1]
        else:
            filename_to_save = img_file.filename

    save_img_dir = os.path.join(config.DB_PATH, filename_to_save)

    # Check for duplicate in MongoDB first
    if mongo_collection.find_one({"filename": filename_to_save}):
        raise HTTPException(status_code=409, detail=f"{filename_to_save} has already in the database (MongoDB record).")

    # Also check if the file already exists on disk (as DeepFace relies on it)
    if os.path.exists(save_img_dir):
        raise HTTPException(status_code=409, detail=f"{save_img_dir} has already in the database (file system).")

    # Save image to database (file system for DeepFace)
    try:
        if (config.RESIZE is False) and (to_gray is False):
            with open(save_img_dir, "wb") as w:
                shutil.copyfileobj(img_file.file, w)
        else:
            image = Image.open(img_file.file)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            if config.RESIZE:
                image = image.resize(config.SIZE)

            np_image = np.array(image)
            np_image = cv.cvtColor(np_image, cv.COLOR_RGB2BGR)

            if to_gray:
                np_image = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

            cv.imwrite(save_img_dir, np_image)

        # Store metadata in MongoDB
        mongo_collection.insert_one({"filename": filename_to_save, "full_path": save_img_dir})

    except Exception as e:
        # If saving fails, clean up any partially written file
        if os.path.exists(save_img_dir):
            os.remove(save_img_dir)
        raise HTTPException(status_code=500, detail=f"Something went wrong when saving the image or updating MongoDB: {e}")
    finally:
        img_file.file.close()
        # image.close() # Image might not always be opened as PIL Image

    remove_representation() # delete all representation_*.pkl created by DeepFace.find
    return {
        "message": f"{filename_to_save} has been saved at {save_img_dir} and registered in MongoDB.",
    }


@app.post("/recognition/")
def face_recognition(
    img_file:UploadFile =  File(...,description="Query image file"),
    to_gray: bool | None = Query(
            default=True,
            description="Whether save image in gray scale or not"),
    return_image_name:bool = Query(default=True, description="Whether return only image name or full image path"),
):

    '''
    Do Face Recognition task, give the image which is
    the most similar with the input image from the
    database - in this case is a folder of images

    Arguments:
        img_file(File): image file
        return_image_name(bool): Decide whether return only image file (img) or image file with extension (img.[jpg|jpeg])
    Return:
        Return path to the most similar image file
    '''

    # Check if database is empty via MongoDB
    if mongo_collection.count_documents({}) == 0:
        return "No image found in the database"

    # Save query image to ./query
    if not os.path.exists("query"):
        os.makedirs("query")

    if '/' in img_file.filename:
        query_img_path = os.path.join("query", img_file.filename.split('/')[-1])
    elif "\\" in img_file.filename:
        query_img_path = os.path.join("query", img_file.filename.split("\\")[-1])
    else:
        query_img_path = os.path.join("query", img_file.filename)

    # Convert image to gray (if necessary) then save it
    if to_gray:
        image = Image.open(img_file.file)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        np_image = np.array(image)
        np_image = cv.cvtColor(np_image, cv.COLOR_BGR2GRAY)

        cv.imwrite(query_img_path, np_image)
    else:
        with open(query_img_path, "wb") as w:
            shutil.copyfileobj(img_file.file, w)

    # Face detection - recognition
    df = DeepFace.find(img_path=query_img_path,
                        db_path = config.DB_PATH,
                        model_name = config.MODELS[config.MODEL_ID],
                        distance_metric = config.METRICS[config.METRIC_ID],
                        detector_backend = config.DETECTORS[config.DETECTOR_ID],
                        silent = True, align = True, prog_bar = False, enforce_detection=False)

    # Remove query image
    os.remove(query_img_path)

    # If faces are detected/recognized
    if not df.empty:
        path_to_img, metric = df.columns
        ascending = True
        if config.METRIC_ID == 0:
            ascending = False
        df = df.sort_values(by=[metric], ascending=ascending)
        value_img_path = df[path_to_img].iloc[0]

        if return_image_name:
            return_value = value_img_path.split(os.path.sep)[-1]
            return_value = return_value.split(".")[0]
            return {
                "result": return_value,
            }
        else:
            return {
                "result": value_img_path,
            }
    else:
        return {
            "result": "No faces have been found"
        }


@app.put('/change-file-name')
def change_img_name(
    src_filename:str = Query(..., description="Filename going to be change (e.g., img1.jpeg)"),
    new_filename:str = Query(..., description="New filename (e.g., im2.jpeg or im2)")
    ):
    '''
    Change file name in database and update MongoDB.

    Arguments:
        src_filename (str) Original filename (e.g: img1.jpeg)
        new_filename (str) New filename (e.g: im2.jpeg or im2)
    Returns:
        images/img1.jpeg -> images/im2.jpeg
    '''

    if mongo_collection.count_documents({}) == 0:
        return "No image found in the database"

    # Construct full paths
    src_full_path = os.path.join(config.DB_PATH, src_filename)

    if "." not in new_filename:
        # If new_filename doesn't have an extension, try to get it from src_filename
        if "." in src_filename:
            extension = src_filename.split(".")[-1]
            new_full_path = os.path.join(config.DB_PATH, new_filename + "." + extension)
            new_filename_with_ext = new_filename + "." + extension
        else: # No extension in either, assume no extension
            new_full_path = os.path.join(config.DB_PATH, new_filename)
            new_filename_with_ext = new_filename
    else:
        new_full_path = os.path.join(config.DB_PATH, new_filename)
        new_filename_with_ext = new_filename


    if not os.path.exists(src_full_path):
        raise HTTPException(status_code=404, detail=f'File {src_filename} not found in file system.')

    if mongo_collection.find_one({"filename": new_filename_with_ext}):
        raise HTTPException(status_code=409, detail=f"{new_filename_with_ext} already exists in MongoDB.")

    if os.path.exists(new_full_path):
        raise HTTPException(status_code=409, detail=f"{new_full_path} already exists in the file system.")

    try:
        # Rename file on disk
        os.rename(src_full_path, new_full_path)
        # Update MongoDB record
        mongo_collection.update_one(
            {"filename": src_filename},
            {"$set": {"filename": new_filename_with_ext, "full_path": new_full_path}}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error renaming file or updating MongoDB: {e}")

    return {
        "message": f"Successfully changed {src_filename} to {new_filename_with_ext}"
    }


@app.delete('/del-single-image')
def del_img(img_filename:str = Query(..., description="Filename of the image need to be deleted (e.g., img1.jpeg)")):
    '''
    Delete single image file in database and from MongoDB.

    Arguments:
        img_filename (str) Filename of the image (e.g: img1.jpeg)
    '''
    if mongo_collection.count_documents({}) == 0:
        return "No image found in the database"

    full_img_path = os.path.join(config.DB_PATH, img_filename)

    if not os.path.exists(full_img_path):
        raise HTTPException(status_code=404, detail=f'File {img_filename} not found in file system!')

    # Delete from MongoDB first
    delete_result = mongo_collection.delete_one({"filename": img_filename})

    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Image {img_filename} not found in MongoDB.")

    # Delete from file system
    os.remove(full_img_path)

    return {
        "message": f"{img_filename} has been deleted from file system and MongoDB!"
    }


@app.delete('/reset-db')
def del_db():
    '''
    Delete all files in database and clear MongoDB collection.
    '''
    if mongo_collection.count_documents({}) == 0:
        return "No image found in the database"

    # Delete all files from the directory
    for file in os.listdir(config.DB_PATH):
        file_path = os.path.join(config.DB_PATH, file)
        if os.path.isfile(file_path): # Ensure it's a file, not a subdirectory
            os.remove(file_path)

    # Clear MongoDB collection
    mongo_collection.delete_many({})

    if len(os.listdir(config.DB_PATH)) == 0 and mongo_collection.count_documents({}) == 0:
        return {
            "message": "All files have been deleted from file system and MongoDB!"
        }
    else:
        raise HTTPException(status_code=500, detail="Something went wrong during database reset.")