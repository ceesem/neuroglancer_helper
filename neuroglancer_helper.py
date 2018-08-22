import neuroglancer
import matplotlib.colors as cl
from collections import Iterable
from IPython.core.display import display, HTML
from urllib.parse import urlparse
from contextlib import contextmanager
from time import sleep
import numpy as np
import copy

def set_neuroglancer_static_source( url ):
    neuroglancer.set_static_content_source(url=url) 

def show_link( url, text='Link'):
    display(HTML('<a href={}>{}</a>'.format(url,text)))

class ng_url_viewer( ):
    '''
    Replicates simple neuroglancer.viewer.Viewer() properties as an unattached state
    '''
    def __init__(self,
                state= neuroglancer.viewer_state.ViewerState( ),
                base=neuroglancer.url_state.default_neuroglancer_url ):
        self.state = copy.deepcopy( state )
        self.base =  base

    def update_base( self, new_base ):
        self.base = new_base

    def url(self):
        return neuroglancer.to_url( self.state, prefix=self.base )

    def link( self, text='URL_Viewer'):
        show_link( self.url(), text=text )

    @contextmanager
    def txn(self):
        yield self.state

    @classmethod
    def from_url( cls, ng_url ):
        state = neuroglancer.parse_url(ng_url)
        base = '{}://{}'.format( urlparse(ng_url).scheme, urlparse(ng_url).netloc )
        return cls( state=state, base=base )

    @classmethod
    def from_viewer( cls, old_viewer ):
        state= viewer.state
        state.showSlices=False
        base = neuroglancer.url_state.default_neuroglancer_url
        return cls( state=state, base=base )

    def __str__(self):
        return self.url()

    def __repr__(self):
        return self.url()

    def _repr_html_(self):
        return '<a href={}>{}</a>'.format(self.url(), 'URL_Viewer')

def get_image_layers( viewer ):
    image_layers = [] 
    for l in viewer.state.layers:
        if l.type is 'image':
            image_layers.append({'name':l.name,
                                 'source':l.source})
    return image_layers

def add_image_layer( viewer, img_source, layer_name='image' ):
    with viewer.txn() as s:
        s.layers.append(name=layer_name,
                        layer=neuroglancer.ImageLayer(source=img_source))
    return viewer

def set_image_layer( viewer, img_source, layer_name='image'):
    if layer_name not in [l.name for l in url_viewer.state.layers]:
        add_image_layer( viewer, img_source, layer_name)
    elif viewer.state.layers[layer_name].type == 'image':
        with viewer.txn() as s:
            s.layers[layer_name].source = img_source
    else:
        print('{} exists and is not an image layer')
    return viewer

def get_segmentation_layers( viewer ):
    seg_layers = [] 
    for l in viewer.state.layers:
        if l.type is 'segmentation':
            seg_layers.append({'name':l.name,
                               'source':l.source})
    return seg_layers

def add_segmentation_layer( viewer, seg_source, layer_name='segmentation'):
    with viewer.txn() as s:
        s.layers.append(name=layer_name,
                        layer=neuroglancer.SegmentationLayer(source=seg_source))
    return viewer

def set_segmentation_layer( viewer, seg_source, layer_name ):
    if layer_name not in [l.name for l in url_viewer.state.layers]:
        add_image_layer( viewer, img_source, layer_name)
    elif viewer.state.layers[layer_name].type == 'segmentation':
        with viewer.txn() as s:
            s.layers[layer_name].source = seg_source
    else:
        print('{} exists and is not an image layer')
    return viewer

def add_selected_objects( viewer, object_ids, layer_name = 'segmentation' ):
    if not isinstance(object_ids, Iterable):
        object_ids = [object_ids]
    with viewer.txn() as s:
        s.layers[layer_name].segments.update(np.int64(object_ids)) 
    return viewer

def get_selected_objects( viewer, layer_name = 'segmentation' ):
    return list( viewer.state.layers[layer_name].segments )

def clear_selection_layer( viewer, layer_name='segmentation' ):
    with viewer.txn() as s:
        s.layers['segmentation'].segments.clear()
    return

def set_annotation_color( viewer, layer_name, color ):
    with viewer.txn() as s:
        s.layers[layer_name].annotationColor = cl.to_hex(color)
    return

def get_annotation_color( viewer, layer_name ):
    return viewer.state.layers[layer_name].annotationColor

def add_annotation_layer( viewer, layer_name, verbose=True):
    if (layer_name in [l.name for l in viewer.state.layers]) & (verbose):
        print('{} is already a layer!'.format(layer_name))
    else:
        with viewer.txn() as s:
            s.layers.append( name=layer_name,
                             layer=neuroglancer.AnnotationLayer() )

        # Neuroglancer seems to take a short delay before layers are ready for objects to be added to them.
        # This sleep adds enough delay to python-bound viewers
        if isinstance(viewer, neuroglancer.viewer.Viewer):
            sleep(2.5)
    return viewer

def get_annotation_class( viewer, layer_names=None, annotation_classes = 'all' ):
    if annotation_classes == 'all':
        annotation_classes = ['point','line','ellipsoid','axis_aligned_bounding_box']

    if layer_names is None:
        layer_names = [l.name for l in viewer.state.layers if l.type == 'annotation']

    annotations=dict()
    for l in viewer.state.layers:
        if (l.type == 'annotation') & (l.name in layer_names):
            annotations[l.name] =  dict( zip(annotation_classes, [[] for i in annotation_classes] ) )
            for row in l.annotations:
                if row.type in annotation_classes:
                    annotations[l.name][row.type].append( row )
    return annotations

def process_point_annotations( point_annotations ):
    xyz = np.array([row.point for row in point_annotations])
    return xyz

def process_line_annotations( line_annotations ):
    xyz0 = np.array([ row.pointA for row in line_annotations ])
    xyz1 = np.array([ row.pointB for row in line_annotations ])
    return xyz0, xyz1

def process_ellipsoid_annotations( ellipsoid_annotations ):
    centers = np.array( [ row.center for row in ellipsoid_annotations ] )
    radii = np.array( [row.radii for row in ellipsoid_annotations] )
    return centers, radii

def process_bounding_box_annotations( bb_annotations ):
    xyz0 = np.array([ row.pointA for row in bb_annotations ])
    xyz1 = np.array([ row.pointB for row in bb_annotations ])
    return xyz0, xyz1

def get_point_annotations( viewer, layer_names = None ):
    """
    Get a dict of pointAnnotationLayers
    """
    points = dict()
    all_point_annos = get_annotation_class( viewer, layer_names=layer_names, annotation_classes=['point'] )
    for l in all_point_annos:
        points[l] = process_point_annotations( all_point_annos[l]['point'] )
    return points 

def add_point_annotations( viewer, xyz, layer_name, color=(1,1,1) ):
    '''
    Add a collection of points to a point annotation layer. This function will create the layer if need be.
    Each point has a unique id. 
    '''
    extant_layers = [l.name for l in viewer.state.layers]
    if layer_name not in extant_layers:
        add_annotation_layer( viewer, layer_name )
        print("Adding layer {}".format(layer_name))

    with viewer.txn() as s:
        s.layers[layer_name].annotationColor = cl.to_hex(color)
        for row in xyz:
            s.layers[layer_name].annotations.append( neuroglancer.PointAnnotation(
                                                       point=np.array(row, dtype=np.int),
                                                       id=neuroglancer.random_token.make_random_token())
                                                    )
    set_annotation_color( viewer, color=color )
    return viewer

def get_line_annotations( viewer, layer_names = None ):
    """
    Get a dict of pointAnnotationLayers
    """
    lines = dict()
    all_line_annos = get_annotation_class( viewer, layer_names=layer_names, annotation_classes=['line'])
    for l in all_line_annos:
        dat = process_line_annotations( all_line_annos[l]['line'] ) 
        lines[l] = {'pointA':dat[0], 'pointB':dat[1]}
    return lines 

def add_line_annotations( viewer, xyz0, xyz1, layer_name, color=(1,1,1) ):
    '''
    Add a collection of points to a point annotation layer. This function will create the layer if need be.
    Each point has a unique id. 
    '''
    extant_layers = [l.name for l in viewer.state.layers]
    if layer_name not in extant_layers:
        add_annotation_layer( viewer, layer_name )
        print("Adding layer {}".format(layer_name))

    with viewer.txn() as s:
        s.layers[layer_name].annotationColor = cl.to_hex(color)
        for ind, xyz0_row in enumerate(xyz0):
            s.layers[layer_name].annotations.append( neuroglancer.LineAnnotation(
                                                       pointA=np.array(xyz0_row, dtype=np.int),
                                                       pointB=np.array(xyz1[ind,:], dtype=np.int),
                                                       id=neuroglancer.random_token.make_random_token())
                                                    )
    set_annotation_color( viewer,layer_name=layer_name, color=color )
    return viewer

def get_ellipsoid_annotations( viewer, layer_names = None ):
    ellipsoids = dict()
    all_ellipsoid_annos = get_annotation_class( viewer, layer_names=layer_names, annotation_classes=['ellipsoid'] )
    for l in all_ellipsoid_annos:
        dat = process_ellipsoid_annotations( all_ellipsoid_annos[l]['ellipsoid'] )
        ellipsoids[l] = {'center':dat[0], 'radii':dat[1]}
    return ellipsoids 

def add_ellipsoid_annotations( viewer, centers, radii, layer_name, color=(1,1,1) ):
    extant_layers = [l.name for l in viewer.state.layers]
    if layer_name not in extant_layers:
        add_annotation_layer( viewer, layer_name )
        print("Adding layer {}".format(layer_name))

    with viewer.txn() as s:
        s.layers[layer_name].annotationColor = cl.to_hex(color)
        for ind, row in enumerate(centers):
            s.layers[layer_name].annotations.append( neuroglancer.EllipsoidAnnotation(
                                                       center=np.array(row, dtype=np.float),
                                                       radii=np.array(xyz1[ind,:], dtype=np.int),
                                                       id=neuroglancer.random_token.make_random_token())
                                                    )
    set_annotation_color( viewer, layer_name=layer_name, color=color )
    return viewer

def get_bounding_box_annotations( viewer, layer_names = None ):
    bounding_boxes = dict()
    all_bounding_box_annos = get_annotation_class( viewer, layer_names=layer_names, annotation_classes=['axis_aligned_bounding_box'] )
    for l in all_bounding_box_annos:
        dat = process_bounding_box_annotations( all_bounding_box_annos[l]['axis_aligned_bounding_box'] ) 
        bounding_boxes[l] = {'pointA':dat[0], 'pointB':dat[1]} 
    return bounding_boxes 

def get_layout( viewer ):
    return viewer.state.layout

def set_layout( viewer, layout ):
    with viewer.txn() as s:
        s.layout = layout
    return viewer

def set_location( viewer, xyz, in_voxels = True ):
    with viewer.txn() as s:
        if in_voxels:
            s.voxel_coordinates = xyz
        else:
            s.voxel_coordinates = np.array( np.array( xyz ) / viewer.state.voxel_size, dtype=np.int)
    return viewer

def in_voxel_coordinates( viewer, xyz ):
    return np.array( np.array( xyz ) / viewer.state.voxel_size, dtype=np.int)