import numpy as np
import vtk


def generate_spike(times, peak_location):
    """Generate a spike waveform"""

    y = np.exp(- 100 * np.abs(times - peak_location)) - \
        0.5 * np.exp(-(times - peak_location - 0.02) ** 2
                     / (2 * (0.02/3) ** 2))

    return y


def display_mesh(vertices, triangles, vertex_data=None):
    """Display a mesh in an interactive window.

    Args:
        vertices: The vertices of the mesh to display. Must be a numpy array
            with a shape of (N, 3) where N is the number of vertices.
        triangles: The triangles of the mesh to display. Must be numpy array
            with a shape of (M, 3) where M is the number of triangles.
        vertex_data: The vertex data used to color the mesh. Must be a numpy
            array with a shape of (N, T) where T is the number of time points.

    """

    # Create a new vtk renderer and rendering window.
    renderer = vtk.vtkRenderer()
    renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
    renderer.SetBackground(0.0, 0.0, 0.0)
    rendering_window = vtk.vtkRenderWindow()
    rendering_window.AddRenderer(renderer)

    # Allow the user to interact with the mesh.
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(rendering_window)
    interactor.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera())

    # Transfer the meshes vertices to the vtk format.
    points = vtk.vtkPoints()
    for vertex in vertices:
        points.InsertNextPoint(*vertex)

    # Create the triangles of the surface.
    vtk_triangles = vtk.vtkCellArray()
    for triangle in triangles:
        vtk_triangle = vtk.vtkTriangle()
        vtk_triangle.GetPointIds().SetId(0, triangle[0])
        vtk_triangle.GetPointIds().SetId(1, triangle[1])
        vtk_triangle.GetPointIds().SetId(2, triangle[2])
        vtk_triangles.InsertNextCell(vtk_triangle)

    # Create the poly data, mapper, and actor.
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(vtk_triangles)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    if vertex_data is not None:

        vtk_color_array = vtk.vtkFloatArray()
        vtk_color_array.SetName('color')
        data = vertex_data
        vtk_color_array.SetNumberOfComponents(data.shape[1])
        for intensity in data:
            vtk_color_array.InsertNextTuple(intensity)

        max = np.abs(data).max()
        if max == 0:
            max = 1.0
        vtk_lut = _two_color_lut(minimum=-max, maximum=max,
                                 start=(1.0, 0.5, 0.5))
        polydata.GetPointData().AddArray(vtk_color_array)
        mapper.SetLookupTable(vtk_lut)
        mapper.SelectColorArray('color')
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SetColorModeToMapScalars()
        mapper.UseLookupTableScalarRangeOn()
        mapper.Update()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Add the actor to the renderer.
    renderer.AddActor(actor)

    current_component = 0

    def _key_press(obj, event):
        """Callback for rendering window keypress.

        Parameters
        ----------
        obj : ?
            Object that triggered the event.

        event : ?
            Object that contains information about the event.

        """

        nonlocal current_component

        # Get the key.
        key = obj.GetKeySym()

        if key == 'Right':

            current_component = np.clip(current_component + 1,
                                        0, data.shape[1])
            vtk_lut.SetVectorComponent(current_component)

        elif key == 'Left':

            current_component = np.clip(current_component - 1,
                                        0, data.shape[1])
            vtk_lut.SetVectorComponent(current_component)

        elif key == 'q':

            # Cleanup when the user closes the window.
            rendering_window.Finalize()
            interactor.TerminateApp()

        # Update the viewer.
        interactor.Render()

    # Add keypress events.
    interactor.AddObserver(
        'KeyPressEvent',
        lambda obj, event: _key_press(obj, event))

    # Start rendering.
    rendering_window.Render()
    interactor.Start()

    # Cleanup when the user closes the window.
    rendering_window.Finalize()
    interactor.TerminateApp()


def _two_color_lut(minimum=-1.0, maximum=1.0,
                   start=None, middle=None, end=None):

    if start is None:
        start = [0.0, 0.0, 1.0]
    if middle is None:
        middle = [0.9, 0.9, 0.9]
    if end is None:
        end = [0.0, 1.0, 0.0]

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(512)
    lut.SetRange(minimum * 0.5, maximum * 0.5)
    for i in reversed(range(256)):
        lut.SetTableValue(
            i,
            (middle[0] - start[0]) * i / 255.0 + start[0],
            (middle[1] - start[1]) * i / 255.0 + start[1],
            (middle[2] - start[2]) * i / 255.0 + start[2],
            1.0)
    for i in range(256):
        lut.SetTableValue(
            256 + i,
            (middle[0] - end[0]) * (255 - i) / 255.0 + end[0],
            (middle[1] - end[1]) * (255 - i) / 255.0 + end[1],
            (middle[2] - end[2]) * (255 - i) / 255.0 + end[2],
            1.0)
    lut.Build()

    return lut
