## Usage

Under project root:

### common:
  - --config_path: path to your degradation pipeline config file
  - --make_video: will make video if raised

### modes:
  1. single: degrades a single video
    - --input_root: the folder *directly* containing frames of the desired video to degrade

    - --output_root: the *parent* folder that wraps the folder containing the output pack

    example:
    ```
    python dataset/degrade_video_albu.py --input_root ../dataset/GOT10/test/GOT-10k_Test_000001/ \
    --output_root data/GOT10/test --make_video --mode single  --config_path dataset/aug_strong.yaml 
    ```

  2. multiple: degrades multiple videos

    - --input_root: the folder containing folders of videos that you want to degrade
    - --output_root: the parent folder that contains the folders of the result videos

    example:
    ```
    python dataset/degrade_video_albu.py --input_root ../dataset/GOT10/test/ \
    --output_root data/GOT10/test --mode multiple --config_path dataset/aug_strong.yaml
    ```

  3. txt: the videos are indicated in a txt file

    - --input_root: the folder that at least contains every folders of videos that you want to degrade
    - --output_root: the parent folder that contains the folders of the result videos
    - --txt: the path to the txt file which indicates the *relative path* from the input_root to the video

    example:
    ```
    python dataset/degrade_video_albu.py --input_root ../dataset/GOT10/test/ \
    --output_root data/GOT10/test --mode txt --txt dataset/test_subset.txt --config_path dataset/aug_strong.yaml
    ```
  

