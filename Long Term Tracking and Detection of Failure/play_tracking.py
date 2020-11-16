import os
import sys
import pdb
import cv2
import pandas as pd



# Flags
comp_flag = 1;


def print_help():
    """
    Prints help.
    """
    print("\nUSAGE:\n  python3 play_tracking.py <mode>",
          "<video path> <csv1 path> <csv2 path>")
    print("\nPARAMETERS:")
    print("  <mode>: Can operate in save or display or compare modes")
    print("          save: saves video with bounding box extracted from <csv1 path>")
    print("          disp: displays video with bounding box extracted from <csv1 path>")
    print("          comp: Compares GT and Tracker using <csv1 path> and <csv2 path>")
    print("  <csv1 path>: Path to csv file having bounding box coordinates")
    print("  <csv2 path>: ground truth bounding box coordinates. Used in 'comp' mode\n")


def check_arguments():
    """
    Checks arguments. Prints help if it detects improper number of
    arguments.
    """
    # Parsing arguments
    if( len(sys.argv) < 2):
        print_help()
        sys.exit(1)
    elif( sys.argv[1] == "save" or sys.argv[1] == "disp" ):
        if( len(sys.argv) != 4 ):
            print_help()
            sys.exit(1)
    else:
        print("else")
        if( len(sys.argv) != 5):
            print_help()
            sys.exit(1)


def get_method(full_path):
    """
    Extracts tracking method name from full
    path of csv file
    """
    csv_name  = os.path.basename(full_path)
    csv_noext = os.path.splitext(csv_name)[0]
    method    = csv_noext.split("__")[2]
    return csv_noext, method


def display_tracking(vr, df, method):
    """
    Plays video with bounding box extracted from\
    csv file.
    """
    fr_suc, fr = vr.read()
    ht, wd, ch = fr.shape
    if not(fr_suc):
        print("ERROR:\n\t Unable to read video file,")
        print("\t",sys.argv[1])
        sys.exit(1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # data frame loop
    for idx,row in df.iterrows():
        poc     = row['poc']
        bbox    = (row['x'], row['y'] ,
                   row['w'], row['h'])
        flag    = row['flag']
        # set video to poc
        vr.set(cv2.CAP_PROP_POS_FRAMES,poc)
        fr_suc, fr = vr.read()
        # Drawing bounding box
        fr_bb   = cv2.rectangle(fr, bbox, (0,255,0), 3)
        txt     = method + ":" + str(flag)
        cv2.putText(fr_bb,txt,(10,450), font, 4,(0,255,255),2,cv2.LINE_AA)
        # Show frame with bounding box
        cv2.imshow("Tracking",fr_bb)
        cv2.waitKey(1)
    # Releasing video capture and writer
    vr.release()
    cv2.destroyAllWindows()


def write_tracking(vr, df, fname, method):
    """
    Plays video with bounding box extracted from\
    csv file.
    """
    fr_suc, fr = vr.read()
    fr_size    = fr.shape
    w          = fr_size[1]
    h          = fr_size[0]
    # ??? <----- Need to change to something else other than MJPG
    vw         = cv2.VideoWriter(fname+".avi",
                                 cv2.VideoWriter_fourcc(*'XVID')
                                 ,30.0, (w,h))
    ht, wd, ch = fr.shape
    if not(fr_suc):
        print("ERROR:\n\t Unable to read video file,")
        print("\t",sys.argv[1])
        sys.exit(1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # data frame loop
    for idx,row in df.iterrows():
        poc     = row['poc']
        print(poc)
        bbox    = (row['x'], row['y'] ,
                   row['w'], row['h'])
        flag    = row['flag']
        # set video to poc
        # vr.set(cv2.CAP_PROP_POS_FRAMES,poc)
        fr_suc, fr = vr.read()
        # Drawing bounding box
        fr_bb   = cv2.rectangle(fr, bbox, (0,255,0), 3)
        txt     = method + ":" + str(flag)
        cv2.putText(fr_bb,txt,(10,450), font, 4,(0,255,255),2,cv2.LINE_AA)
        vw.write(fr_bb)
    # Releasing video capture and writer
    vr.release()
    vw.release()


def compare_tracking(vr, df1, dfgt, method):
    """
    Shows ground truth bounding box in green and
    tracked in red.
        df1 = GT
        df2 = Algorithm
    """
    # Message for Users
    print("Key strokes : ")
    print("\t s = Successful tracking")
    print("\t f = Failed tracking")
    print("\t q = Quit and save generated ground truth")
    fr_suc, fr = vr.read()
    ht, wd, ch = fr.shape
    if not(fr_suc):
        print("ERROR:\n\t Unable to read video file,")
        print("\t",sys.argv[1])
        sys.exit(1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Groudn truth data frame loop
    ks  = []
    pocs= []
    xgt = []
    ygt = []
    wgt = []
    hgt = []
    x   = []
    y   = []
    w   = []
    h   = []
    k = ord('s')
    for idx,row in dfgt.iterrows():
        if( k == ord('q') and idx != 0 ):
            break

        poc     = row['poc']

        # Bounding box from ground truth
        bboxgt  = ( row['x'], row['y'] ,
                    row['w'], row['h'] )


        # Bounding box from tracker
        if(idx == 0):
            bboxtr = bboxgt
            flag   = 1
        else:
            df1_row = df1[df1['poc'] == poc]
            bboxtr  = ( df1_row['x'].values[0], df1_row['y'].values[0], df1_row['w'].values[0], df1_row['h'].values[0] )
            flag    = df1_row['flag'].values[0]

        # set video to poc
        vr.set(cv2.CAP_PROP_POS_FRAMES,poc)
        fr_suc, fr = vr.read()

        # Drawing bounding box
        fr_gt_bb   = cv2.rectangle(fr, bboxgt, (0,255,0), 3)
        fr_bbs     = cv2.rectangle(fr_gt_bb, bboxtr, (0,0,255), 3)
        txt        = method + ":" + str(flag)
        cv2.putText(fr_bbs,txt, (10,350), font, 2, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(fr_bbs, "Ground Truth", (10,400), font, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(fr_bbs, "Tracking method", (10,450), font, 1, (0,0,255), 2, cv2.LINE_AA)

        # Show frame with bounding box
        cv2.imshow("Tracking",fr_bbs)
        k = cv2.waitKey(0) % 256

        while(1):
            if ( k == ord('s') ) or ( k == ord('f') ):
                pocs    = pocs + [poc]
                ks = ks + [str(chr(k))]
                xgt = xgt + [row['x']]
                ygt = ygt + [row['y']]
                wgt = wgt + [row['w']]
                hgt = hgt + [row['h']]
                x   = x   + [bboxtr[0]]
                y   = y   + [bboxtr[1]]
                w   = w   + [bboxtr[2]]
                h   = h   + [bboxtr[3]]
                break
            elif( k == ord('q')):
                break
            else:
                print("Invalid key, Try again")
                k = cv2.waitKey(0) % 256

    # Releasing video capture and writer
    vr.release()
    cv2.destroyAllWindows()

    # Save as dataframe
    df = pd.DataFrame(columns=['poc','xgt','ygt','wgt','hgt','x','y','w','h','label'])
    df['poc'] = pocs
    df['x']   = x
    df['y']   = y
    df['w']   = w
    df['h']   = h
    df['xgt']   = xgt
    df['ygt']   = ygt
    df['wgt']   = wgt
    df['hgt']   = hgt
    df['label'] = ks
    df.to_csv("gt_Vs_nxcorr_Video2.csv",index=False)



check_arguments()
mode          = sys.argv[1]
vr            = cv2.VideoCapture(sys.argv[2])
df1           = pd.read_csv(sys.argv[3])
if (mode == "disp"):
    fname, method = get_method(sys.argv[3])
    display_tracking(vr, df1, method)
elif (mode == "save"):
    fname, method = get_method(sys.argv[3])
    write_tracking(vr,df1, fname, method)
elif (mode == "comp"):
    fname, method = get_method(sys.argv[4])
    dfgt = pd.read_csv(sys.argv[4])
    compare_tracking(vr,df1,dfgt, method)
else:
    print("ERROR: Invalid operation selected")
    print_help()
    sys.exit(1)
