js = """
<script>
    navigator.geolocation.getCurrentPosition(
        function(position) {
            const params = new URLSearchParams(window.location.search);
            if (!params.has('location')) {
                params.set('location', position.coords.latitude + "," + position.coords.longitude);
                window.location.search = params.toString();
            }
        },
        function(error) {
            if (error.code == error.PERMISSION_DENIED)
                alert("You denied the request for Geolocation.");
        }
    );
</script>
"""
