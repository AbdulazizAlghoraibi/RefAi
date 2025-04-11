import cv2 
import numpy as np
from utils.video_utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer






def main():
    # read video
    video_frames = read_video('input/08fd33_4.mp4')  

    #initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=[True],
                                       stub_path='stubs/track_stubs.pkl')
    

    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    
    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    
    
    '''print("===== Debugging tracks['ball'] =====")
    for frame_num, ball_data in enumerate(tracks['ball']):
        print(f"Frame {frame_num}: {ball_data}")
    print("=====================================")'''



    # Assign Players Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team] 
    
    

    


    
    
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks['ball'])

    
    
    
    
    
    #draw output
    ## draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

   
   

   
    # save video
    save_video(output_video_frames, 'output_videos/output_video_final.avi')

if __name__ == "__main__":
    main()    