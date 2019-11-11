import vtk
from pynput.keyboard import Key, Controller
from vtk.util import numpy_support
import pickle
import os
from os import listdir
from os.path import isfile, join
import time
import csv
import numpy as np

filter = vtk.vtkWindowToImageFilter()
renderWindowInteractor = vtk.vtkRenderWindowInteractor()

iteration = 0

matrix_iter = 0

camera = vtk.vtkCamera()

position = -10

keyboard = Controller()


def close_window(iren):
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()


def keypressCallback(obj, ev):
    key = obj.GetKeySym()
    if key == "s":
        filter.Modified()
        filter.SetInputBufferTypeToZBuffer()
        filter.Update()

        scale = vtk.vtkImageShiftScale()
        scale.SetOutputScalarTypeToUnsignedChar()
        scale.SetInputConnection(filter.GetOutputPort())
        scale.SetShift(0)
        scale.SetScale(-300)

        scale2 = vtk.vtkImageShiftScale()
        scale2.SetOutputScalarTypeToUnsignedChar()
        scale2.SetInputConnection(scale.GetOutputPort())
        scale2.SetShift(255)

        global iteration
        iteration = iteration + 1

        file_name = "/home/vishnusanjay/cmpt764/Final project/last_examples/depthimages_1/chair_" + \
                    str(iteration) + ".bmp"
        imageWriter = vtk.vtkBMPWriter()
        imageWriter.SetFileName(file_name)
        imageWriter.SetInputConnection(scale2.GetOutputPort())
        imageWriter.Write()


def main():
    colors = vtk.vtkNamedColors()

    #     dir = "/home/vishnusanjay/cmpt764/Final project/last_examples/objs"

    #     onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    #     print(onlyfiles)

    counterrr = 0

    onlyfiles = []

    mtl_files = []

    with open('/home/vishnusanjay/cmpt764/Final project/04256520.csv', 'rt', encoding="utf8") as f:
        reader = csv.reader(f)
    your_list = list(reader)

    list_arr = np.asarray(your_list)
    print(len(your_list))
    print(list_arr[:, 2])

    folder_list = []

    for i in range(len(your_list)):
        if list_arr[i, 2].find("chair") != -1:
            print(list_arr[i, 0].split('.')[1])
            folder_list.append(list_arr[i, 0].split('.')[1])

    print(counterrr)
    print(onlyfiles)
    print(mtl_files)
    # exit(1)

    for i in range(len(onlyfiles)):
        global position
        position = -5
        print(onlyfiles[i])
        print(mtl_files[i])
        # if onlyfiles[i].find("1a38407b3036795d19fb4103277a6b93") ==-1:
        #     continue

        # file_name = join(dir,onlyfiles[i])
        reader = vtk.vtkOBJReader()
        reader.SetFileName(onlyfiles[i])
        # reader.SetFileNameMTL(mtl_files[i])
        reader.Update()

        counter = 0
        while (counter < 2):
            counter = counter + 1
            print(position)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(colors.GetColor3d("red"))

            # # Visualize

            camera = vtk.vtkCamera()
            camera.SetFocalPoint(0, 0, 0)
            camera.SetPosition(10, 0, 0)
            camera.SetViewUp(0, 1, 0)
            camera.SetPosition(20, 2, position)

            position = position + 10

            leftVP = [0.0, 0.0, 0.5, 1.0]
            rightVP = [0.5, 0.0, 1.0, 1.0]

            renderer = vtk.vtkRenderer()
            # renderer.SetViewport(leftVP)
            renderer.SetActiveCamera(camera)

            renderWindow = vtk.vtkRenderWindow()
            renderWindow.SetWindowName("Polygon")
            renderWindow.AddRenderer(renderer)
            renderWindow.SetSize(512, 512)
            renderWindowInteractor = vtk.vtkRenderWindowInteractor()
            renderWindowInteractor.SetRenderWindow(renderWindow)

            renderer.AddActor(actor)
            renderer.SetBackground(255, 255, 255)
            renderWindow.Render()

            filter.SetInput(renderWindow)
            # filter.SetViewport(leftVP)
            filter.SetInputBufferTypeToZBuffer()
            # filter.Update()
            # filter.Modified()

            scale = vtk.vtkImageShiftScale()
            scale.SetOutputScalarTypeToUnsignedChar()
            scale.SetInputConnection(filter.GetOutputPort())
            scale.SetShift(0)
            scale.SetScale(-255)

            scale2 = vtk.vtkImageShiftScale()
            scale2.SetOutputScalarTypeToUnsignedChar()
            scale2.SetInputConnection(scale.GetOutputPort())
            scale2.SetShift(255)

            # depthMapper = vtk.vtkImageMapper()
            # depthMapper.SetInputConnection(scale.GetOutputPort())
            # # depthMapper.SetColorLevel(1)

            # depthActor = vtk.vtkActor2D()
            # depthActor.SetMapper(depthMapper)

            # depthRenderer = vtk.vtkRenderer()
            # depthRenderer.SetViewport(rightVP)
            # depthRenderer.AddActor2D(depthActor)
            # depthRenderer.SetBackground(255,255,255)
            # renderWindow.AddRenderer(depthRenderer)

            # filter.Modified()

            # i=0
            # while i <2 :
            #     i = i+1
            #     filter.Modified()
            modelviewMatrix = renderWindow.GetRenderers().GetFirstRenderer().GetActiveCamera().GetModelViewTransformMatrix()
            projectionMatrix = vtk.vtkCamera().GetProjectionTransformMatrix(
                renderWindow.GetRenderers().GetFirstRenderer())
            mvp = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Multiply4x4(projectionMatrix, modelviewMatrix, mvp)

            temp = [0] * 16  # the matrix is 4x4
            mvp.DeepCopy(temp, mvp)
            global matrix_iter
            matrix_iter = matrix_iter + 1
            save_file = file_name = "/home/vishnusanjay/cmpt764/Final project/last_examples/depthimages/chair_" + str(
                matrix_iter) + ".p"

            pickle.dump(temp, open(save_file, "wb"))

            # imageWriter = vtk.vtkBMPWriter()
            # imageWriter.SetFileName("/home/vishnusanjay/cmpt764/Final project/last_examples/ChairBasic2/shortChair01.bmp")
            # imageWriter.SetInputConnection(scale.GetOutputPort())
            # imageWriter.Write()
            # time.sleep(1)

            keyboard.press('r')
            keyboard.press('s')

            time.sleep(1)

            keyboard.press('q')

            # renderWindowInteractor.AddObserver('LeftButtonPressEvent', UpdateFilter)
            # renderWindowInteractor.AddObserver('RightButtonPressEvent', saveimg)
            renderWindowInteractor.AddObserver('KeyPressEvent', keypressCallback)
            # renderWindowInteractor.AddObserver("LeftButtonReleaseEvent", UpdateFilter)
            renderWindowInteractor.Initialize()
            renderWindowInteractor.Start()


if __name__ == '__main__':
    main()
