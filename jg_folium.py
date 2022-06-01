import folium, io
from PIL import Image

def MapForPoints( points, zoom_start=8 ):
    a,b=points.min()
    c,d=points.max()
    start_coords = ((c-a)/2,(d-b)/2)
    folium_map = folium.Map(location=start_coords, zoom_start=zoom_start)
    folium_map.fit_bounds( ((a,b), (c,d)) )
    return folium_map

default_marker = lambda lat,lon,**kwargs : folium.CircleMarker((lat,lon),**kwargs)
default_describer = lambda lat,lon : str((lat,lon))

def MarkPointsOnMap( points, marker=default_marker, describe=default_describer, **kwargs ):
    folium_map = MapForPoints( points )
    for lat,lon in points.values:
        marker(lat,lon,popup=describe(lat,lon),**kwargs).add_to(folium_map)
    return folium_map

def FoliumToPng( folium_map, file_name, rendering_seconds=5 ):
    img_data = folium_map._to_png(rendering_seconds)
    img = Image.open(io.BytesIO(img_data))
    img.save(file_name+'.png')