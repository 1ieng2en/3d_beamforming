from pcd_aligner import PointCloud_PreProcessor


# this is the postprocessing section for the scanned point cloud
# use KinfuScanner/kinfu_example.exe to scan
# scanned file will stored in the same folder with name: kf_output.ply

while True:
    freq_input = input("Do you want start from a raw scan? (y/[n]")
    if freq_input.lower() == 'n':
        print("omit pre-processing, now start point cloud post-processing")
        break  
    processor = PointCloud_PreProcessor('/KinfuScanner/')
    processor.pcd_read()
    processor.pcd_crop()
    processor.pcd_write()

