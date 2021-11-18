import os
import subprocess


def toa_correct_dataset(params):
    print('IN TOA CORRECT')
    data_path = params.project_path + "data/raw/SPARCS_dataset/"
    data_toa_path = params.project_path + "data/processed/SPARCS_TOA/"
    data_output_path = params.project_path + "data/output/SPARCS/"
    products = sorted(os.listdir(data_path))
    products = [p for p in products if 'data.tif' in p]
    print('IN TOA CORRECT 1, length', len(products))
    c1 = 'C:/Users/Dewire/.conda/envs/RStest/Scripts/gdal_merge.py'
    cf1 = 'C:/Users/Dewire/.conda/envs/RStest/Scripts/fmask_usgsLandsatMakeAnglesImage.py'

    cf2 = 'C:/Users/Dewire/.conda/envs/RStest/Scripts/fmask_usgsLandsatSaturationMask.py'
    cf3 = 'C:/Users/Dewire/.conda/envs/RStest/Scripts/fmask_usgsLandsatTOA.py'
    c2 = '-separate'
    c3 = 'HFA'
    c4 = 'COMPRESSED=YES'

    # REMOVED _18 BAD FILE
    for product in products:
        print(product)
        p2 = data_path[:-1] + '/' + product[0:24] + '_data.tif'
        p1 = 'gdal_translate'
        # Save individual bands temporarily

        for b in range(1, 11):
            if b < 8:
                p3 = data_output_path + product[0:25] + 'B' + str(b) + '.tif'
                cmd = [p1, '-b', str(b), p2, p3]
                print(cmd)
                subprocess.run(cmd)
                print('IN TOA CORRECT 2')

            else:
                band_number = b + 1
                p3 = data_output_path + product[0:25] + 'B' + str(band_number) + '.tif'
                cmd = [p1, '-b', str(b), p2, p3]
                print(cmd)
                subprocess.run(cmd)
                print('IN TOA CORRECT 3')
        #
        # # ## CREATE LIST of FILE NAME FRO GDAL_MERGE
        subprocess.run('dir /b /s *B?.tif > list.txt', shell=True, cwd=data_output_path)
        print('IN TOA CORRECT 41')
        subprocess.run('dir /b /s *B10.tif *B11.tif> list2.txt', shell=True, cwd=data_output_path)
        print('IN TOA CORRECT 51')

        # # Run Fmask -GDAL
        c5 = data_output_path[:-1] + '/ref.img'
        c6 = data_output_path[:-1] + '/list.txt'
        cmd = ['python', c1, c2, '-of', c3, '-co', c4, '-o', c5, '--optfile', c6]
        subprocess.run(cmd)
        print('IN TOA CORRECT 4')

        c5 = data_output_path[:-1] + '/thermal.img'
        c6 = data_output_path[:-1] + '/list2.txt'
        cmd = ['python', c1, c2, '-of', c3, '-co', c4, '-o', c5, '--optfile', c6]
        subprocess.run(cmd)
        print('IN TOA CORRECT 5')

        # Run Fmask
        d1 = data_path[:-1] + '/' + product[0:21] + '_mtl.txt'
        d2 = data_output_path[:-1] + '/ref.img'
        d3 = data_output_path[:-1] + '/angles.img'
        cmd = [cf1, '-m', d1, '-t', d2, '-o', d3]
        cmd = [cf1, '-m', d1, '-t', d2, '-o', d3]
        subprocess.run(cmd)
        print('IN TOA CORRECT 6')

        d1 = data_path[:-1] + '/' + product[0:21] + '_mtl.txt'
        d2 = data_output_path[:-1] + '/ref.img'
        d3 = data_output_path[:-1] + '/saturationmask.img'
        cmd = [cf2, '-i', d2, '-m', d1, '-o', d3]
        subprocess.run(cmd)
        print('IN TOA CORRECT 7')

        d1 = data_path[:-1] + '/' + product[0:21] + '_mtl.txt'
        d2 = data_output_path[:-1] + '/ref.img'
        d3 = data_output_path[:-1] + '/angles.img'
        d4 = data_output_path[:-1] + '/toa.img'

        cmd = [cf3, '-i', d2, '-m', d1, '-z', d3, '-o', d4]
        subprocess.run(cmd)

        print('IN TOA CORRECT 8')

        # Translate to .tif file
        p3 = data_toa_path[:-1] + '/' + product[0:24] + '_toa.TIF'
        cmd = [p1, '-of', 'Gtiff', d4, p3]
        subprocess.run(cmd)
        print('IN TOA CORRECT 9')

        # Delete temp files
        subprocess.run('del *.tif *.img *.xml *.txt', shell=True, cwd=data_output_path)
        print('IN TOA CORRECT 10')

