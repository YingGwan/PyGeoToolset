import numpy as np
import math
from scipy.linalg import lu_factor, lu_solve
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.animation import FFMpegWriter

import time
import threading
from threading import Timer
import importlib 
import xlrd
from datetime import date,datetime
import open3d as o3d
import colorBar
import shutil
import copy
import trimesh as tm

dis_MAX = 0.015    #unit: m -> 0.02m -> 20mm

BasisTableSize = int(200)
fittedSurfacePointSize = 30    #100x100

########################################################################################################################################
###########################################Boundary Extraction############################################################################
########################################################################################################################################
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                    # R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                  # axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)




########################################################################################################################################
###########################################Open3D Visualizer############################################################################
########################################################################################################################################
class Open3DVisualizer:
    """This is open3D visualizer for mesh rendering"""
    #Input just need point cloud, and calcualte edge list and face list \
    #would be handled inside this class
    
    #self.mesh -> real-time fitted surface
    
    def CreateWindow(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
       
    
    #real-time point cloud
    def GetRawPointCloud(self,pc):
        self.pcd_rawdata = pc
        
    
    #####################################
    ####Ctrl Pnt Grid Visualization######
    #####################################
    def GetCtrlPointCloud(self, ctrlPC,ctrlPntNumber):
        self.pcd_ctrlpnt =  ctrlPC
        self.pcd_ctrlpnt_number_line = ctrlPntNumber
        
    def CalculateEdgeList(self):
    
        #get numpy array
        self.ctrlgrid_pointTable = np.asarray(self.pcd_ctrlpnt.points)
        #print(self.pcd_ctrlpntArray)
        
        cols=int((self.pcd_ctrlpnt_number_line))
        edgeNum = 2* (cols-1)*(cols)
        #print("Edgenum is %d"%(edgeNum))
        self.ctrlgrid_edgeTable = np.zeros((edgeNum,2))         #used to render
        self.edgeTable = np.arange(edgeNum*2, dtype='int32')    #old code
        
        iterSum = cols
        firstpara = int(0)
        for i in range(cols):
            for j in range(cols - 1):
                self.edgeTable[i*(cols-1)*2 + j*2] = firstpara + j * cols
                self.edgeTable[i*(cols-1)*2 + j*2 + 1] = firstpara + (j+1)*cols
            firstpara=firstpara+1
        
        firstpara = 0
        
        for i in range(iterSum,iterSum*2,1):
            for j in range(cols - 1):
                self.edgeTable[i*(cols-1)*2 + j*2] = firstpara+j
                self.edgeTable[i*(cols-1)*2 + j*2 + 1] = firstpara + j+1
            firstpara=firstpara+cols
        
        #print("EdgeTable: \n",self.edgeTable,"\n")
        
        for i in range(edgeNum):
            self.ctrlgrid_edgeTable[i][0] = self.edgeTable[i*2+0]
            self.ctrlgrid_edgeTable[i][1] = self.edgeTable[i*2+1]
        
        #print("EdgeTable: \n",self.edgeTable,"\n")
        
        self.ctrlpnt_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(self.ctrlgrid_pointTable),
        lines=o3d.utility.Vector2iVector(self.ctrlgrid_edgeTable))
        
        self.ctrlpnt_line_set.paint_uniform_color([0,1,0]) 
        
        
    def VisualizeCtrlGrid(self):
        o3d.visualization.draw_geometries([self.pcd_ctrlpnt,self.ctrlpnt_line_set])
        #o3d.visualization.draw_geometries([self.ctrlpnt_line_set])

    def VisualizeCtrlGrid_RawData(self):
        #o3d.visualization.draw_geometries([self.pcd_ctrlpnt,self.ctrlpnt_line_set])
        self.pcd_rawdata.paint_uniform_color((0,0,0))
        self.pcd_ctrlpnt.paint_uniform_color((1,0,0))
        o3d.visualization.draw_geometries([self.pcd_ctrlpnt,self.ctrlpnt_line_set,self.pcd_rawdata])
    
    #####################################
    #Visualization of RT Fitted Surface##
    #####################################
    
    #fitted surface contains only points which are Nx3 array
    def GetFittedPointArray(self,fittedSurface, edgePntNumber):
        self.fittedPntSet = fittedSurface
        self.fittedPntEdgePntNumber = edgePntNumber
    
    def CalculateFittedSurface(self):
        cols=int((self.fittedPntEdgePntNumber))
        faceNum = 2* (cols-1)*(cols-1)
        iterSum = (cols-1)*(cols-1)
        
        self.faceTable = np.arange(faceNum*3, dtype='int32')    #old code
        self.faceArray = np.zeros((faceNum,3))
        firstPara = 0
        
        faceIdx= 0
        for i in range(iterSum):
            if (i%(cols-1) ==0)and(i!=0):
                firstPara = firstPara+1
            self.faceTable[i*6+0] = firstPara
            self.faceTable[i*6+1] = firstPara+cols
            self.faceTable[i*6+2] = firstPara+1
            
            self.faceTable[i*6+3] = firstPara+cols
            self.faceTable[i*6+4] = firstPara+cols+1
            self.faceTable[i*6+5] = firstPara+1
            
            self.faceArray[faceIdx][0] = firstPara
            self.faceArray[faceIdx][1] = firstPara+cols
            self.faceArray[faceIdx][2] = firstPara+1
            
            self.faceArray[faceIdx+1][0] = firstPara+cols
            self.faceArray[faceIdx+1][1] = firstPara+cols+1
            self.faceArray[faceIdx+1][2] = firstPara+1
            
            firstPara = firstPara+1
            faceIdx = faceIdx +2
        
        # From numpy to Open3D        
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(self.fittedPntSet)
        self.mesh.triangles = o3d.utility.Vector3iVector(self.faceArray)
        
        self.mesh.compute_vertex_normals()
        #self.mesh.paint_uniform_color([0,0,0]) 
        
        #o3d.visualization.draw_geometries([self.mesh])
        
    ########################################
    #Visualization of Target Fitted Surface##
    ########################################
    
    #target fitted surface contains only points which are Nx3 array
    def GetTargetFittedPointArray(self,fittedSurface, edgePntNumber):
        self.targetfittedPntSet = fittedSurface
        self.targetfittedPntEdgePntNumber = edgePntNumber
    
    def CalculateTargetFittedSurface(self):
        cols=int((self.targetfittedPntEdgePntNumber))
        faceNum = 2* (cols-1)*(cols-1)
        iterSum = (cols-1)*(cols-1)
        
        self.targetfaceTable = np.arange(faceNum*3, dtype='int32')    #old code
        self.targetfaceArray = np.zeros((faceNum,3))
        firstPara = 0
        
        faceIdx= 0
        for i in range(iterSum):
            if (i%(cols-1) ==0)and(i!=0):
                firstPara = firstPara+1
            self.targetfaceTable[i*6+0] = firstPara
            self.targetfaceTable[i*6+1] = firstPara+cols
            self.targetfaceTable[i*6+2] = firstPara+1
            
            self.targetfaceTable[i*6+3] = firstPara+cols
            self.targetfaceTable[i*6+4] = firstPara+cols+1
            self.targetfaceTable[i*6+5] = firstPara+1
            
            self.targetfaceArray[faceIdx][0] = firstPara
            self.targetfaceArray[faceIdx][1] = firstPara+cols
            self.targetfaceArray[faceIdx][2] = firstPara+1
            
            self.targetfaceArray[faceIdx+1][0] = firstPara+cols
            self.targetfaceArray[faceIdx+1][1] = firstPara+cols+1
            self.targetfaceArray[faceIdx+1][2] = firstPara+1
            
            firstPara = firstPara+1
            faceIdx = faceIdx +2
        
        # From numpy to Open3D        
        self.targetmesh = o3d.geometry.TriangleMesh()
        self.targetmesh.vertices = o3d.utility.Vector3dVector(self.targetfittedPntSet)
        self.targetmesh.triangles = o3d.utility.Vector3iVector(self.targetfaceArray)
        
        self.targetmesh.compute_vertex_normals()
    
    def BackupTargetMesh(self):
        self.targetmesh_default = copy.deepcopy(self.targetmesh)
    
    #apply transformation matrix (4x4) to target mesh
    def ApplyRT_TargetMesh(self, fileName):
        #RT = np.loadtxt(fileName,delimiter = ',')
        data_second_trans = np.loadtxt(fileName,dtype = np.float,delimiter = ' ')
        Rotation = data_second_trans[0:3,:]
        Translation = data_second_trans[3,:]
        #print("rotation: \n",Rotation)
        #print("Translation:\n",Translation)
        
        Transformation = np.zeros((4,4))
        Transformation[3,3] = 1.0
        for i in range(3):
            for j in range(3):
                Transformation[i,j] = Rotation[i,j]

        Transformation[0,3] = Translation[0]
        Transformation[1,3] = Translation[1]
        Transformation[2,3] = Translation[2]
        print("Transformation: \n",Transformation)
        self.targetmesh = copy.deepcopy(self.targetmesh_default)
        self.targetmesh.transform(Transformation)
        
    #apply inverse oof transformation matrix (4x4) to target mesh
    def AdjustToOriginalPos(self):
        #RT = np.loadtxt(fileName,delimiter = ',')
        
        transformation_file_prefix = "./Target/RT/Transformation"
        transformation_file_middle = str(0)
        transformation_file_postfix = ".txt"

        fileName = transformation_file_prefix+transformation_file_middle+transformation_file_postfix
        
        data_second_trans = np.loadtxt(fileName,dtype = np.float,delimiter = ' ')
        Rotation = data_second_trans[0:3,:]
        Translation = data_second_trans[3,:]
        #print("rotation: \n",Rotation)
        #print("Translation:\n",Translation)
        
        Transformation = np.zeros((4,4))
        Transformation[3,3] = 1.0
        for i in range(3):
            for j in range(3):
                Transformation[i,j] = Rotation[i,j]

        Transformation[0,3] = Translation[0]
        Transformation[1,3] = Translation[1]
        Transformation[2,3] = Translation[2]
        
        
        inv_Transformation = np.linalg.inv(Transformation)
        print("Inverse of Transformation: \n",inv_Transformation)
        self.targetmesh_default.transform(inv_Transformation)
        
    def ReadRawPointCloud(self,fileName):
        self.rtMesh = o3d.io.read_triangle_mesh(fileName)
        
        
    def ApplyColorToRTMesh(self, selection = 1):
        "selection == 0: point-to-point distance"
        "selection == 1: point-to-mesh  distance"
        "selection == 2: point-to-mesh  distance: raw point cloud to target fitted surface"
        if selection == 0:
            self.meshVerSet = np.asarray(self.mesh.vertices)
            self.targetmeshVerSet = np.asarray(self.targetmesh.vertices)
            self.meshVerSetLen = self.meshVerSet.shape[0]
            
            self.pcdRealTime = o3d.geometry.PointCloud()
            self.pcdRealTime.points = o3d.utility.Vector3dVector(self.meshVerSet)
            
            self.pcdTarget = o3d.geometry.PointCloud()
            self.pcdTarget.points = o3d.utility.Vector3dVector(self.targetmeshVerSet)
            
            dists = self.pcdRealTime.compute_point_cloud_distance(self.pcdTarget)
            dists = np.asarray(dists) 
            
            np_colors = np.zeros((self.meshVerSetLen,3))
            for i in range(self.meshVerSetLen):
                [rr,gg,bb] = colorBar._changeValueToColor(dis_MAX,0,dists[i])
                np_colors[i][0] = rr
                np_colors[i][1] = gg
                np_colors[i][2] = bb
                if i % 100 == 0:
                    print("color cal progress: %lf percent\n"%(i/self.meshVerSetLen*100.0))
            
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(np_colors)
        
        
        
        #######################
        #new code:
        if selection == 1:
            queryMesh = tm.Trimesh(np.asarray(self.targetmesh.vertices),np.asarray(self.targetmesh.triangles))
            sourcePointSet = np.asarray(self.mesh.vertices)
            sourcePointSetLen = sourcePointSet.shape[0]
            [closest, dists, triangle_id] = tm.proximity.closest_point(queryMesh, sourcePointSet)
           
            np_colors = np.zeros((sourcePointSetLen,3))
            for i in range(sourcePointSetLen):
                [rr,gg,bb] = colorBar._changeValueToColor(dis_MAX,0,dists[i])
                np_colors[i][0] = rr
                np_colors[i][1] = gg
                np_colors[i][2] = bb
                if i % 100 == 0:
                    print("color cal progress: %lf percent\n"%(i/sourcePointSetLen*100.0))
            
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(np_colors)
            
            
        #new code:
        if selection == 2:
            self.rtMesh.compute_vertex_normals()
        
            queryMesh = tm.Trimesh(np.asarray(self.targetmesh.vertices),np.asarray(self.targetmesh.triangles))
            sourcePointSet = np.asarray(self.rtMesh.vertices)
            sourcePointSetLen = sourcePointSet.shape[0]
            [closest, dists, triangle_id] = tm.proximity.closest_point(queryMesh, sourcePointSet)
           
            np_colors = np.zeros((sourcePointSetLen,3))
            for i in range(sourcePointSetLen):
                [rr,gg,bb] = colorBar._changeValueToColor(dis_MAX,0,dists[i])
                np_colors[i][0] = rr
                np_colors[i][1] = gg
                np_colors[i][2] = bb
                if i % 100 == 0:
                    print("color cal progress: %lf percent\n"%(i/sourcePointSetLen*100.0))
            
            self.rtMesh.vertex_colors = o3d.utility.Vector3dVector(np_colors)
        
    def ShowMesh(self,typeNum, imageIndex = 0):
    
        #When type = :
        #0 -> only real-time mesh
        #1 -> only target mesh
        #2 -> mixture
        #3 -> only real-time with colormap
        #4 -> only real-time raw meshed with colormap
        
        if typeNum == 0:
            o3d.visualization.draw_geometries([self.mesh])
        
        if typeNum == 1:
            o3d.visualization.draw_geometries([self.targetmesh])
            
        if typeNum == 2:
            self.mesh.paint_uniform_color([0,1,0]) 
            self.targetmesh.paint_uniform_color([1,1,0]) 
            o3d.visualization.draw_geometries([self.mesh,self.targetmesh])
            #o3d.visualization.draw_geometries([self.targetmesh])
            
            
        if typeNum == 3:
            #o3d.visualization.draw_geometries([self.mesh])
            edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=False)
            points=o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
            lines=o3d.utility.Vector2iVector(np.asarray(edges))
            colors = [[0, 0, 0] for i in range(np.asarray(self.mesh.vertices).shape[0])]            
            line_mesh1 = LineMesh(points, lines, colors, radius=0.0015)
            line_mesh1_geoms = line_mesh1.cylinder_segments
            o3d.visualization.draw_geometries([*line_mesh1_geoms, self.mesh])
            
        if typeNum == 4:
            self.vis.clear_geometries()
            #o3d.visualization.draw_geometries([self.mesh])
            edges = self.rtMesh.get_non_manifold_edges(allow_boundary_edges=False)
            points=o3d.utility.Vector3dVector(np.asarray(self.rtMesh.vertices))
            lines=o3d.utility.Vector2iVector(np.asarray(edges))
            colors = [[0, 0, 0] for i in range(np.asarray(self.rtMesh.vertices).shape[0])]            
            line_mesh1 = LineMesh(points, lines, colors, radius=0.0005)
            line_mesh1_geoms = line_mesh1.cylinder_segments
            #o3d.visualization.draw_geometries([*line_mesh1_geoms, self.rtMesh])    
            
            #new visualization method
            self.vis.add_geometry(self.rtMesh)
            
            
            line_mesh1.add_line(self.vis)
            
            
           
            
            
            #update geometry
            # Updates
            ##self.vis.update_geometry()
            self.ctr = self.vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters("./leftButton.json")
            self.ctr.convert_from_pinhole_camera_parameters(param)
            self.vis.poll_events()
            self.vis.update_renderer()
            
            
            #
            # self.vis.run()  # user changes the view and press "q" to terminate
            # param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            # o3d.io.write_pinhole_camera_parameters("leftButton.json", param)
            # self.vis.destroy_window()
            time.sleep(5)
            self.vis.capture_screen_image("./fig/color - "+str(imageIndex)+" .png")
            
########################################################################################################################################
###########################################BSpline Basis Function#######################################################################
########################################################################################################################################
class BSplineBasis:
    """This is the class for BSpline basis function"""
    def GetCtrlPntNumber(self,ctrlNumber):
        self.ctrlPntNumber = ctrlNumber
        
    def GetCtrlPntNumber_CtrlOrder(self,ctrlNumber,ctrlOrder):
        self.ctrlPntNumber = ctrlNumber
        self.ctrlOrder = ctrlOrder
        self.knotVecLen = self.ctrlOrder+self.ctrlPntNumber
        self.knotDistance = self.knotVecLen - 2*self.ctrlOrder+1
           
    def PreComputeCombinationNumber(self): 
        #compute combination number
        self.knotVec = np.zeros((self.knotVecLen+1))
        for i in range(1,self.ctrlOrder+1): #[1, self.ctrlOrder]
            #print("i is %d"%(i))
            self.knotVec[i] = 0.0
            self.knotVec[self.ctrlOrder+self.ctrlPntNumber-i+1] = 1.0
        
        N = 1.0
        for i in range(self.ctrlOrder+1,self.ctrlPntNumber+1):    #[self.ctrlOrder+1,self.ctrlPntNumber+1)
            self.knotVec[i] = 1.0 * N / self.knotDistance
            N = N + 1.0
            
    def PrintCombinationNumber(self):
        print("***********************************")
        for i in range(1,self.knotVecLen+1):
            print("%d's knot value: %lf\n"%(i,self.knotVec[i]))
        print("\n\n***********************************")
        
    #use this function to get bspline basis function    
    def Init(self,ctrlNumber,ctrlOrder):
        self.GetCtrlPntNumber_CtrlOrder(ctrlNumber,ctrlOrder)
        self.PreComputeCombinationNumber()
        
   
###########################################################
    def CalBasisValue(self, i, k, t):
        
        left = 0.0
        right = 0.0
        if (t < 0.0) or (t > 1.0):
            print("t is not in range [0,1]...")
            os._exit(-1)
        
        if (i == 0):
            print("i should not be 0...")
            os._exit(-1)
        
        if k==1:
            if (t>=self.knotVec[i]) and (t<self.knotVec[i+1]):
                return 1.0
            else:
                return 0.0
        else:
            leftCoeff = self.CalBasisValue(i,k-1,t)
            rightCoeff = self.CalBasisValue(i+1,k-1,t)
            
            
            if np.fabs(leftCoeff)<0.0000001:
                left = 0.0
            else:
                left = leftCoeff * (t-self.knotVec[i])/(self.knotVec[i+k-1]-self.knotVec[i])
            
            if np.fabs(rightCoeff)<0.0000001:
                right = 0.0
            else:
                right = rightCoeff * (self.knotVec[i+k]-t)/(self.knotVec[i+k]-self.knotVec[i+1])
     
        return (left+right)
        
        
    
    #说明：
    # B_(i,k,t) 其实i是从1开始，到ctrlpntnumber结束, t是从0到1
    # 我们将self.table初始化为二维数组，然后每一行代表一个i，第一行空出来
    #    为了与 i从1开始保持一致
    # 每一纵列都是离散化的从0到1的数 [0,999]
    #interval = 1.0/ (BasisTableSize-1)
    
    def StoreBasisValueIntoFile(self):
        self.table = np.zeros((self.ctrlPntNumber+1, BasisTableSize))
        parameterInterval = 1.0/(BasisTableSize-1)
        for i in range(1,self.ctrlPntNumber+1):
            for j in range(BasisTableSize):
                para = 0.0
                if j == (BasisTableSize-1):
                    para = 0.9999
                else:
                    para = j*parameterInterval
                value = self.CalBasisValue(i,self.ctrlOrder,para)    
                self.table[i][j] = value
        #np.save("BasisTable.npy",self.table)
        np.savetxt("BasisTable.txt",self.table,fmt='%.7lf',delimiter = ',')
    
    def ReadBasisValueFromFile(self):
        #self.table = np.load("BasisTable.npy")
        self.table = np.loadtxt("BasisTable.txt",delimiter=',')
        #print("self.table: \n",self.table)
    
    #i from 1
    def CalBasisValueWithTable(self,i,k,t):
        row=0
        col=0
        
        row = i
        parameterInterval = 1.0/(BasisTableSize-1)
        col = int(t/parameterInterval)
        return self.table[row][col]
    
########################################################################################################################################
###########################################BSpline Surface Fitting Class################################################################
########################################################################################################################################
#Variable Meaning:
# markerPosMat -> marker 3d coordinate (3 X N)

#input: directly read from obj file
#


class BSpline:
    "This is the class for BSpline surface fitting, including computation and visualization"
    def GetCtrlPntNumber_CtrlOrder(self,ctrlNumberx,ctrlNumbery,ctrlOrderx,ctrlOrdery):
        self.ctrlPntNumber_x = ctrlNumberx
        self.ctrlPntNumber_y = ctrlNumbery
        self.ctrlOrder_x = ctrlOrderx
        self.ctrlOrder_y = ctrlOrdery 
        
        
        self.basis_x = BSplineBasis()
        self.basis_y = BSplineBasis()
        
        self.basis_x.Init(self.ctrlPntNumber_x,self.ctrlOrder_x)
        self.basis_y.Init(self.ctrlPntNumber_y,self.ctrlOrder_y)
        
    def ReadObjFile(self,fileName):
        #current folder : ./  
        #parent folder: ../
        
        # print("prepare to open file: ",fileName)
        # fileMesh = o3d.io.read_triangle_mesh(fileName)      
        # fileMesh_pc  =  np.asarray(fileMesh.vertices)   

        #print(fileMesh_pc)
        
        # self.pcd = o3d.geometry.PointCloud()
        # self.pcd.points = o3d.utility.Vector3dVector(fileMesh_pc)
        self.mesh = o3d.io.read_triangle_mesh(fileName)   
        self.pointArray  =  np.asarray(self.mesh.vertices)  
        
        #self.pcd = o3d.io.read_point_cloud(fileName)    
        
        #self.pointArray  = np.asarray(self.pcd.points)
        self.pointArrayLen = self.pointArray.shape[0]
        
        # self.pcd.paint_uniform_color([0,0,0])
        # o3d.visualization.draw_geometries([self.pcd],zoom=0.960,
                                  # front=[0.84841049853050299, -0.048488852712833749, 0.52711332476595241],
                                  # lookat=[-0.00025060799999999939, -0.0041934506499999996, 0.024576648150000002],
                                  # up=[-0.11260970406923929, 0.95646830601522326, 0.26923490512525977])
    
        self.markerPosMat = np.transpose(self.pointArray)
        self.markerNumber = self.pointArrayLen
        
        #print("Marker Position:\n",self.markerPosMat)
        #print("marker number is \n%d"%(self.markerNumber))
        
        
        
    def ShowPC(self):
        o3d.visualization.draw_geometries([self.pcd],zoom=0.960,
                                  front=[0.84841049853050299, -0.048488852712833749, 0.52711332476595241],
                                  lookat=[-0.00025060799999999939, -0.0041934506499999996, 0.024576648150000002],
                                  up=[-0.11260970406923929, 0.95646830601522326, 0.26923490512525977])
    
    def Parameterize(self):
        #print("In parameterization function")
        # Find bounding box
        # self.x_max = np.amax(self.markerPosMat,axis=1)[0]
        # self.x_min = np.amin(self.markerPosMat,axis=1)[0]
        
        # self.y_max = np.amax(self.markerPosMat,axis=1)[1]
        # self.y_min = np.amin(self.markerPosMat,axis=1)[1]
        
        self.x_max = np.amax(self.markerPosMat[0,:])
        self.x_min = np.amin(self.markerPosMat[0,:])
        
        self.y_max = np.amax(self.markerPosMat[1,:])
        self.y_min = np.amin(self.markerPosMat[1,:])
        
        print("y range: ",self.y_max-self.y_min)
        print("x range: ",self.x_max-self.x_min)
    
        #print(self.x_max,self.x_min)
        #print(self.y_max,self.y_min)
        
        #Parameterize
        self.ParameterMat=np.zeros([2,self.markerNumber])
        for i in range(self.markerNumber):
            uu = (self.markerPosMat[0,i] - self.x_min)/(self.x_max - self.x_min)
            vv = (self.markerPosMat[1,i] - self.y_min)/(self.y_max - self.y_min)
            self.ParameterMat[0,i]=uu
            self.ParameterMat[1,i]=vv
            #print("%d: (%lf,%lf)"%(i,self.ParameterMat[0][i],vv))

    def FromJtoJ1J2(self, j):
        j1 = int(j / self.ctrlPntNumber_x+1)
        j2 = int(int(j)%self.ctrlPntNumber_x+1)
        #print(j1,",",j2)
        return np.array([j1,j2])
        
    def FormMatA(self):
        self.MatA = np.zeros([self.markerNumber,self.ctrlPntNumber_x * self.ctrlPntNumber_y])
        for i in range(self.markerNumber):
            for j in range(self.ctrlPntNumber_x * self.ctrlPntNumber_y):
                j1j2=self.FromJtoJ1J2(j)
                # if i == 0:
                    # print("j:%d -> (%lf,%lf)"%(j,j1j2[0],j1j2[1]))
                #value = self.basis_x.CalBasisValue(j1j2[0], self.ctrlOrder_x,self.ParameterMat[0][i]) * self.basis_y.CalBasisValue(j1j2[1], self.ctrlOrder_y,self.ParameterMat[1][i])
                
                value = self.basis_x.CalBasisValueWithTable(j1j2[0], self.ctrlOrder_x,self.ParameterMat[0][i]) * self.basis_x.CalBasisValueWithTable(j1j2[1], self.ctrlOrder_y,self.ParameterMat[1][i])
                
             
                #svalue = 
                
                self.MatA[i,j] = value
            if i % 300 == 0:
                print("Form MatA percent: %lf"%(i/self.markerNumber*100.0))
        
        # Lu Solver Precomputing:
        self.lu, self.piv = lu_factor(np.matmul(self.MatA.transpose(),self.MatA))
    
    
    def FormMatB(self):
        #tempArray = np.array()
        self.MatB = np.zeros([self.markerNumber,3])
        for i in range(self.markerNumber):
            #if i == 0:
            #    self.MatB = np.array([[self.markerPosMat[0][i],self.markerPosMat[1][i],self.markerPosMat[2][i]]])
            #else:
            #    self.MatB = np.insert(self.MatB , 0, values=np.array([self.markerPosMat[0][i],self.markerPosMat[1][i],self.markerPosMat[2][i]]), axis=0)
            self.MatB[i,0]=self.markerPosMat[0][i]
            self.MatB[i,1]=self.markerPosMat[1][i]
            self.MatB[i,2]=self.markerPosMat[2][i]
        
        print("MatB shape value is ->..\n",self.MatB.shape)
        
    def Solve(self):
        print(".................................\n        Begin to solve AX=B\n.................................\n")
        self.x = lu_solve((self.lu, self.piv), np.matmul(self.MatA.transpose(),self.MatB))
        #print("Ctrl Pnt Pos: ->\n",self.x)
        #print("Marker Number: %d\nCtrl Pnt Num: (%d,%d)\nNorm of AX-B: %f\n"%(self.markerNumber, self.ctrlPntNumber_x, self.ctrlPntNumber_y,np.linalg.norm((np.matmul(self.MatA,self.x)-self.MatB))))
    
    def BSplineSolve(self):
        self.Parameterize()
        self.FormMatA()
        self.FormMatB()
        self.Solve()
        print("Solve Finished")
        
    def CalculatePntOnSurface(self,u,v):
        xx = 0
        yy = 0
        zz = 0
        for i in range(self.ctrlPntNumber_x * self.ctrlPntNumber_y):
            j1j2 = self.FromJtoJ1J2(i)
            xx = xx + (self.basis_x.CalBasisValueWithTable(j1j2[0],self.ctrlOrder_x,u)) * (self.basis_x.CalBasisValueWithTable(j1j2[1],self.ctrlOrder_y,v)) * (self.x[i][0])
            yy = yy + (self.basis_x.CalBasisValueWithTable(j1j2[0],self.ctrlOrder_x,u)) * (self.basis_x.CalBasisValueWithTable(j1j2[1],self.ctrlOrder_y,v)) * (self.x[i][1])
            zz = zz + (self.basis_x.CalBasisValueWithTable(j1j2[0],self.ctrlOrder_x,u)) * (self.basis_x.CalBasisValueWithTable(j1j2[1],self.ctrlOrder_y,v)) * (self.x[i][2])
        return np.array([xx,yy,zz])
        
    def GenerateCtrlPointCloud(self):
        self.pcd_ctrlpoint = o3d.geometry.PointCloud()
        self.pcd_ctrlpoint.points = o3d.utility.Vector3dVector(self.x) 
        
        self.pcd_ctrlpoint.paint_uniform_color([0, 0.0, 0]) 
        
        
        
    def GenerateFittedPatch(self):
        "generate fitted surface point cloud"
        
        self.fittedsurfacePntSet = np.zeros((fittedSurfacePointSize*fittedSurfacePointSize,3))
        
        intervalSize = 1.0/ (fittedSurfacePointSize-1)
        idx = 0
        for i in range(fittedSurfacePointSize):
            for j in range(fittedSurfacePointSize):
                u = intervalSize*i
                v = intervalSize*j
                if i == fittedSurfacePointSize-1:
                    u =  u - 0.000001
                if j == fittedSurfacePointSize - 1:
                    v =  v - 0.000001
                
                pnt = self.CalculatePntOnSurface(u,v)
                self.fittedsurfacePntSet[idx][0] = pnt[0]
                self.fittedsurfacePntSet[idx][1] = pnt[1]
                self.fittedsurfacePntSet[idx][2] = pnt[2]
                
                idx = idx +1
                if idx%300 == 0:
                    print("generate fitted surface: %lf percent"%(idx/(fittedSurfacePointSize*fittedSurfacePointSize)*100.0))
        
        #For temporarily usage
        self.pcd_fittedSurface= o3d.geometry.PointCloud()
        self.pcd_fittedSurface.points = o3d.utility.Vector3dVector(self.fittedsurfacePntSet)
        # o3d.visualization.draw_geometries([self.pcd_ctrl_grid],zoom=0.960,
                                  # front=[0.84841049853050299, -0.048488852712833749, 0.52711332476595241],
                                  # lookat=[-0.00025060799999999939, -0.0041934506499999996, 0.024576648150000002],
                                  # up=[-0.11260970406923929, 0.95646830601522326, 0.26923490512525977])
    
    
    ##################################################
    ################Target fitted surface#############
    ##################################################
    #from target control patch to fitted surface
    #input: target control patch
    #output: target fitted patch point set
    
    
    def ReadTargetCtrlFile(self,fileName):
        self.pcd_targetctrl = o3d.io.read_point_cloud(fileName)
        self.pcd_targetctrl.paint_uniform_color([0,0,0])
        self.targetctrlArray  = np.asarray(self.pcd_targetctrl.points)
        self.targetctrlArray_len = self.targetctrlArray.shape[0]
        
        #print(self.targetctrlArray)
        
        self.x = np.zeros((self.targetctrlArray_len,3))
        for i in range(self.targetctrlArray_len):
            self.x[i][0] = self.targetctrlArray[i][0]
            self.x[i][1] = self.targetctrlArray[i][1]
            self.x[i][2] = self.targetctrlArray[i][2]
            
        #"generate fitted surface point cloud"
        
        self.targetfittedsurfacePntSet = np.zeros((fittedSurfacePointSize*fittedSurfacePointSize,3))
        
        intervalSize = 1.0/ (fittedSurfacePointSize-1)
        idx = 0
        for i in range(fittedSurfacePointSize):
            for j in range(fittedSurfacePointSize):
                u = intervalSize*i
                v = intervalSize*j
                if i == fittedSurfacePointSize-1:
                    u =  u - 0.000001
                if j == fittedSurfacePointSize - 1:
                    v =  v - 0.000001
                
                pnt = self.CalculatePntOnSurface(u,v)
                self.targetfittedsurfacePntSet[idx][0] = pnt[0]
                self.targetfittedsurfacePntSet[idx][1] = pnt[1]
                self.targetfittedsurfacePntSet[idx][2] = pnt[2]
                
                idx = idx +1
                if idx%300 == 0:
                    print("generate fitted surface: %lf percent"%(idx/(fittedSurfacePointSize*fittedSurfacePointSize)*100.0))
        
    
    def VisualizeOnePc(self,pc):
        o3d.visualization.draw_geometries([pc],zoom=0.960,
                                  front=[0.84841049853050299, -0.048488852712833749, 0.52711332476595241],
                                  lookat=[-0.00025060799999999939, -0.0041934506499999996, 0.024576648150000002],
                                  up=[-0.11260970406923929, 0.95646830601522326, 0.26923490512525977])

    
    def VisualizePCs(self,pc1,pc2):
        pc1.paint_uniform_color([0, 0.0, 0]) 
        pc2.paint_uniform_color([242/255, 71/255, 71/255]) 
        o3d.visualization.draw_geometries([pc1+pc2],zoom=0.960,
                                  front=[0.84841049853050299, -0.048488852712833749, 0.52711332476595241],
                                  lookat=[-0.00025060799999999939, -0.0041934506499999996, 0.024576648150000002],
                                  up=[-0.11260970406923929, 0.95646830601522326, 0.26923490512525977])
                                  
                                  
    def CalculateTargetCtrlEdgeList(self):
    
        #get numpy array  pcd_ctrlpoint
        #self.ctrlgrid_pointTable = self.targetctrlArray
        #print(self.pcd_ctrlpntArray)
        
        cols=int((self.ctrlPntNumber_x))
        edgeNum = 2* (cols-1)*(cols)
        #print("Edgenum is %d"%(edgeNum))
        self.ctrlgrid_edgeTable = np.zeros((edgeNum,2))         #used to render
        self.edgeTable = np.arange(edgeNum*2, dtype='int32')    #old code
        
        iterSum = cols
        firstpara = int(0)
        for i in range(cols):
            for j in range(cols - 1):
                self.edgeTable[i*(cols-1)*2 + j*2] = firstpara + j * cols
                self.edgeTable[i*(cols-1)*2 + j*2 + 1] = firstpara + (j+1)*cols
            firstpara=firstpara+1
        
        firstpara = 0
        
        for i in range(iterSum,iterSum*2,1):
            for j in range(cols - 1):
                self.edgeTable[i*(cols-1)*2 + j*2] = firstpara+j
                self.edgeTable[i*(cols-1)*2 + j*2 + 1] = firstpara + j+1
            firstpara=firstpara+cols
        
        #print("EdgeTable: \n",self.edgeTable,"\n")
        
        for i in range(edgeNum):
            self.ctrlgrid_edgeTable[i][0] = self.edgeTable[i*2+0]
            self.ctrlgrid_edgeTable[i][1] = self.edgeTable[i*2+1]
        
        #print("EdgeTable: \n",self.edgeTable,"\n")
        
        self.ctrlpnt_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(self.targetctrlArray),
        lines=o3d.utility.Vector2iVector(self.ctrlgrid_edgeTable))
        
        self.ctrlpnt_line_set.paint_uniform_color([0,1,0]) 
    
    
    
    def CalculateCtrlEdgeList(self):
    
        #get numpy array  pcd_ctrlpoint
        self.ctrlgrid_pointTable = np.asarray(self.pcd_ctrlpoint.points)
        #print(self.pcd_ctrlpntArray)
        
        cols=int((self.ctrlPntNumber_x))
        edgeNum = 2* (cols-1)*(cols)
        #print("Edgenum is %d"%(edgeNum))
        self.ctrlgrid_edgeTable = np.zeros((edgeNum,2))         #used to render
        self.edgeTable = np.arange(edgeNum*2, dtype='int32')    #old code
        
        iterSum = cols
        firstpara = int(0)
        for i in range(cols):
            for j in range(cols - 1):
                self.edgeTable[i*(cols-1)*2 + j*2] = firstpara + j * cols
                self.edgeTable[i*(cols-1)*2 + j*2 + 1] = firstpara + (j+1)*cols
            firstpara=firstpara+1
        
        firstpara = 0
        
        for i in range(iterSum,iterSum*2,1):
            for j in range(cols - 1):
                self.edgeTable[i*(cols-1)*2 + j*2] = firstpara+j
                self.edgeTable[i*(cols-1)*2 + j*2 + 1] = firstpara + j+1
            firstpara=firstpara+cols
        
        #print("EdgeTable: \n",self.edgeTable,"\n")
        
        for i in range(edgeNum):
            self.ctrlgrid_edgeTable[i][0] = self.edgeTable[i*2+0]
            self.ctrlgrid_edgeTable[i][1] = self.edgeTable[i*2+1]
        
        #print("EdgeTable: \n",self.edgeTable,"\n")
        
        self.ctrlpnt_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(self.ctrlgrid_pointTable),
        lines=o3d.utility.Vector2iVector(self.ctrlgrid_edgeTable))
        
        self.ctrlpnt_line_set.paint_uniform_color([0,1,0]) 
        
def saveCtrlPointArray(destFile,npArray3n):
    #np.save(destFile, npArray3n)
    np.savetxt(destFile, npArray3n, delimiter=' ')   # X is an array
        
        
        
if __name__ == "__main__":
    
    useRawPntCld = True

    basis = BSplineBasis()
    basis.Init(10,4)             #
    
    #activate this function if BasisTableSize has been changed
    basis.StoreBasisValueIntoFile()
    ###
    TargetBSpline = BSpline()
    TargetBSpline.GetCtrlPntNumber_CtrlOrder(10,10,4,4)
    TargetBSpline.basis_x.ReadBasisValueFromFile()
    
    TargetBSpline.ReadTargetCtrlFile("targetCtrlPnt.xyz")           #read target ctrl patch and generate fitted surface

    #translate from numpy array to o3d pc
    TargetBSpline.CalculateTargetCtrlEdgeList()
    
    o3d.visualization.draw_geometries([TargetBSpline.ctrlpnt_line_set,TargetBSpline.pcd_targetctrl])
    #ctrlpnt_line_set
    
    
   
    
    
    # visualizer = Open3DVisualizer()
    # visualizer.CreateWindow()
    
    # visualizer.GetTargetFittedPointArray(targetSpline.targetfittedsurfacePntSet,fittedSurfacePointSize )
    # visualizer.CalculateTargetFittedSurface()
    # visualizer.BackupTargetMesh()
    
    # visualizer.AdjustToOriginalPos()
    
  
            
