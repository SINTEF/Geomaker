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
    map.fitBounds(drawnItems.getLayer(lfid).getBounds());
}

function initialize() {
    // Interface to OpenStreetMap and Google Maps
    var osmUrl = 'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
    var osmAttrib = '&copy; <a href="http://openstreetmap.org/copyright">OpenStreetMap</a> contributors';
    var googleUrl = 'http://www.google.com/maps/vt?lyrs=p@189&gl=cn&x={x}&y={y}&z={z}';
    var googleAttrib = '&copy; Google';

    var osm = L.tileLayer(osmUrl, { maxZoom: 18, attribution: osmAttrib });
    var google = L.tileLayer(googleUrl, { maxZoom: 18, attribution: googleAttrib });

    // Actual map object with drawn items
    map = new L.Map('map', { center: new L.LatLng(63.43011, 10.39478), zoom: 15 });
    drawnItems = L.featureGroup().addTo(map);

    // Control for choosing which layers to show
    L.control.layers({
        'OpenStreetMap': L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map),
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

        map.on('draw:created', function (event) {
            event.layer.setStyle(unselectedStyle);
            drawnItems.addLayer(event.layer);
            if (typeof Interface != 'undefined') {
                var json = JSON.stringify(event.layer.toGeoJSON());
                Interface.add_poly(event.layer._leaflet_id, json);
            }
        });

        map.on('draw:edited', function (event) {
            if (typeof Interface != 'undefined') {
                event.layers.eachLayer(function (layer) {
                    var json = JSON.stringify(layer.toGeoJSON());
                    Interface.edit_poly(layer._leaflet_id, json);
                })
            }
        });

        map.on('draw:deleted', function (event) {
            if (typeof Interface != 'undefined') {
                event.layers.eachLayer(function (layer) {
                    Interface.remove_poly(layer._leaflet_id);
                })
            }
        });

        var moving = false;

        map.on('movestart', function (event) { moving = true; })
        map.on('moveend', function (event) { moving = false; })

        map.on('mouseup', function (event) {
            if (typeof Interface != 'undefined' && !moving) {
                Interface.select_poly(-1);
            }
        });

        drawnItems.on('click', function (event) {
            if (typeof Interface != 'undefined') {
                Interface.select_poly(event.layer._leaflet_id);
            }
        });
    });
}
