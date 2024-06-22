from main import recommender, all_features, image_paths_list, model

# Change file path to outfit you would like to receive recommendations on!
input_image_path = '/Users/benpentecost/Documents/CodingProjects/PythonProjects/ClothingRecommenationTool/women fashion/strapless, sequined dress that sparkles with multiple colors.jpg'
recommender(input_image_path, all_features, image_paths_list, model, top_n=4)