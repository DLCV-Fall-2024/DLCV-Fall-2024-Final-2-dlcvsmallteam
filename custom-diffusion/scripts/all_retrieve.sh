rm -rf real_reg/*

python src/retrieve_new.py --target_name "cat" --outpath "real_reg/cat2"
python src/retrieve_new.py --target_name "dog" --outpath "real_reg/dog6"

python src/retrieve_new.py --target_name "flower" --outpath "real_reg/flower_1"
python src/retrieve_new.py --target_name "vase" --outpath "real_reg/vase"

python src/retrieve_new.py --target_name "dog" --outpath "real_reg/dog"
python src/retrieve_new.py --target_name "cat" --outpath "real_reg/pet_cat1"

python src/retrieve_new.py --target_name "wearable glasses" --outpath "real_reg/wearable_glasses"
python src/retrieve_new.py --target_name "watercolor" --outpath "real_reg/watercolor"