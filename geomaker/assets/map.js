var map, drawnItems, selectedLayer;

var selectedStyle = {
    color: "#ff0000",
};

var unselectedStyle = {
    color: "#0000ff",
};

function add_object(points) {
    var retval;
    L.geoJson({
        type: "Feature",
        properties: {},
        geometry: {"type": "Polygon", "coordinates": [points]},
    }).eachLayer(function (layer) {
        layer.setStyle(unselectedStyle);
        drawnItems.addLayer(layer);
        retval = layer._leaflet_id;
    });
    return retval;
}

function select_object(lfid) {
    if (typeof selectedLayer != 'undefined') {
        selectedLayer.setStyle(unselectedStyle);
        selectedLayer = undefined;
    }
    if (lfid >= 0) {
        var layer = drawnItems.getLayer(lfid);
        layer.setStyle(selectedStyle);
        selectedLayer = layer;
    }
}

function focus_object(lfid) {
    if (lfid >= 0) {
        map.fitBounds(drawnItems.getLayer(lfid).getBounds());
    }
    else if (drawnItems.getLayers().length > 0) {
        map.fitBounds(drawnItems.getBounds());
    }
}

function remove_object(lfid) {
    drawnItems.removeLayer(lfid);
}

function initialize() {
    // Actual map object with drawn items
    map = new L.Map('map', {
        center: new L.LatLng(63.43011, 10.39478),
        zoom: 15,
        doubleClickZoom: false,
    });
    drawnItems = L.featureGroup().addTo(map);

    // Control for choosing which layers to show
    L.control.layers({
        'Kartverket Topo4': L.tileLayer(
            'https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=topo4&zoom={z}&x={x}&y={y}'
        ).addTo(map),
        'Kartverket Grunnkart': L.tileLayer(
            'https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=norges_grunnkart&zoom={z}&x={x}&y={y}'
        ),
        'Kartverket Terreng': L.tileLayer(
            'https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=terreng_norgeskart&zoom={z}&x={x}&y={y}'
        ),
        'Karverket Gr\u00E5': L.tileLayer(
            'https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=norges_grunnkart_graatone&zoom={z}&x={x}&y={y}'
        ),
        'Kartverket Enkelt': L.tileLayer(
            'https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=egk&zoom={z}&x={x}&y={y}'
        ),
        'Norge i bilder': L.tileLayer('https://kartverket.maplytic.no/tile/_nib/{z}/{x}/{y}.jpeg'),
        'OpenStreetMap': L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'),
        'OpenTopoMap': L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png'),
        'Google Roads': L.tileLayer('http://mt0.google.com/maps/vt?lyrs=m@189&gl=cn&x={x}&y={y}&z={z}'),
        'Google Terrain': L.tileLayer('http://mt0.google.com/maps/vt?lyrs=p@189&gl=cn&x={x}&y={y}&z={z}'),
        'Google Satellite': L.tileLayer('http://mt0.google.com/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}'),
        'Google Hybrid': L.tileLayer('http://mt0.google.com/maps/vt?lyrs=y@189&gl=cn&x={x}&y={y}&z={z}'),
        'Google Shaded': L.tileLayer('http://mt0.google.com/maps/vt?lyrs=t@189&gl=cn&x={x}&y={y}&z={z}'),
    }, {
        'Drawn areas': drawnItems,
    }, {
        position: 'bottomleft',
        collapsed: true,
    }).addTo(map);

    // Full draw control: allowing drawing and editing
    var drawFull = new L.Control.Draw({
        edit: {
            featureGroup: drawnItems,
        },
        draw: {
            polyline: false,
            marker: false,
            circlemarker: false,
            circle: false,
        },
    }).addTo(map);

    // Communicate new, edited and deleted layers to Python via QWebChannel
    new QWebChannel(qt.webChannelTransport, function (channel) {
        var Interface = channel.objects.Interface;

        var moving = false;
        var deselect = false;

        map.on('draw:created', function (event) {
            deselect = false;
            event.layer.setStyle(unselectedStyle);
            drawnItems.addLayer(event.layer);
            if (typeof Interface != 'undefined') {
                var json = JSON.stringify(event.layer.toGeoJSON());
                Interface.emit('polygon_added', event.layer._leaflet_id, json);
            }
        });

        map.on('draw:edited', function (event) {
            deselect = false;
            if (typeof Interface != 'undefined') {
                event.layers.eachLayer(function (layer) {
                    var json = JSON.stringify(layer.toGeoJSON());
                    Interface.emit('polygon_edited', layer._leaflet_id, json);
                })
            }
        });

        map.on('draw:deleted', function (event) {
            if (typeof Interface != 'undefined') {
                event.layers.eachLayer(function (layer) {
                    Interface.emit('polygon_deleted', layer._leaflet_id);
                })
            }
        });

        map.on('movestart', function (event) { moving = true; })
        map.on('moveend', function (event) { moving = false; })

        map.on('mouseup', function (event) {
            if (typeof Interface != 'undefined' && !moving) {
                deselect = true;
                setTimeout(function () {
                    if (deselect) {
                        Interface.emit('polygon_selected', -1);
                    }
                }, 10);
            }
        });

        drawnItems.on('click', function (event) {
            if (typeof Interface != 'undefined') {
                Interface.emit('polygon_selected', event.layer._leaflet_id);
                deselect = false;
            }
        });

        drawnItems.on('dblclick', function (event) {
            if (typeof Interface != 'undefined') {
                Interface.emit('polygon_double_clicked', event.layer._leaflet_id);
                deselect = false;
            }
        })
    });
}
