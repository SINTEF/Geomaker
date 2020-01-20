var map, drawnItems;

function add_object(name, points) {
    new QWebChannel(qt.webChannelTransport, function (channel) {
        window.MainWindow = channel.objects.Main;

        L.geoJson({
            type: "Feature",
            properties: {},
            geometry: {"type": "Polygon", "coordinates": [points]},
        }).eachLayer(function (layer) {
            drawnItems.addLayer(layer);
            if (typeof MainWindow != 'undefined') {
                MainWindow.connect(name, layer._leaflet_id);
            }
        });
    });
}

function initialize() {
    // Interface to OpenStreetMap and Google Maps
    var osmUrl = 'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
    var osmAttrib = '&copy; <a href="http://openstreetmap.org/copyright">OpenStreetMap</a> contributors';
    var googleUrl = 'http://www.google.com/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}';
    var googleAttrib = '&copy; Google';

    var osm = L.tileLayer(osmUrl, { maxZoom: 18, attribution: osmAttrib });
    var google = L.tileLayer(googleUrl, { maxZoom: 18, attribution: googleAttrib });

    // Actual map object with drawn items
    map = new L.Map('map', { center: new L.LatLng(63.43011, 10.39478), zoom: 15 });
    drawnItems = L.featureGroup().addTo(map);

    // Control for choosing which layers to show
    L.control.layers({
        'OpenStreetMap': osm.addTo(map),
        'Google': google,
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
        window.MainWindow = channel.objects.Main;

        map.on('draw:created', function (event) {
            drawnItems.addLayer(event.layer);
            if (typeof MainWindow != 'undefined') {
                var json = JSON.stringify(event.layer.toGeoJSON());
                MainWindow.add_object(event.layer._leaflet_id, json);
            }
        });

        map.on('draw:edited', function (event) {
            if (typeof MainWindow != 'undefined') {
                event.layers.eachLayer(function (layer) {
                    var json = JSON.stringify(layer.toGeoJSON());
                    MainWindow.edit_object(layer._leaflet_id, json);
                })
            }
        });

        map.on('draw:deleted', function (event) {
            if (typeof MainWindow != 'undefined') {
                event.layers.eachLayer(function (layer) {
                    MainWindow.remove_object(layer._leaflet_id);
                })
            }
        });
    });
}
