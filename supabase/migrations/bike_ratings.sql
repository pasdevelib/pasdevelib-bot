-- Table pour les notes de velos individuels
CREATE TABLE IF NOT EXISTS bike_ratings (
  id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  station_id  text NOT NULL,
  bike_id     text NOT NULL,
  stars       int  NOT NULL CHECK (stars BETWEEN 1 AND 5),
  bike_type   text DEFAULT 'unknown',
  user_id     uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  rated_at    timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS bike_ratings_bike_id_idx ON bike_ratings(bike_id);
CREATE INDEX IF NOT EXISTS bike_ratings_station_id_idx ON bike_ratings(station_id);
CREATE UNIQUE INDEX IF NOT EXISTS bike_ratings_bike_user_idx
  ON bike_ratings(bike_id, user_id)
  WHERE user_id IS NOT NULL;

ALTER TABLE bike_ratings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "public_read" ON bike_ratings FOR SELECT USING (true);
CREATE POLICY "authenticated_insert" ON bike_ratings FOR INSERT
  TO authenticated WITH CHECK (auth.uid() = user_id OR user_id IS NULL);
CREATE POLICY "service_role_all" ON bike_ratings USING (true) WITH CHECK (true);
