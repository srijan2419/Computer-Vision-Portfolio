"""
Script to:
1. Clean up production data with duration < 5 minutes for 2025-11-13
2. Reassign sequential IDs to all tires in order
"""
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "instance" / "tire_production.db"

if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 80)
print("PART 1: CLEANUP 2025-11-13 PRODUCTION DATA")
print("=" * 80)

# Query tires for 2025-11-13
cursor.execute("""
    SELECT id, start_time, end_time, duration_seconds, status 
    FROM tire_production 
    WHERE date(created_at) = '2025-11-13'
    ORDER BY created_at
""")

tires_nov13 = cursor.fetchall()
print(f"\nFound {len(tires_nov13)} tires on 2025-11-13:\n")

tires_to_delete_nov13 = []
for tire_id, start_time, end_time, duration_seconds, status in tires_nov13:
    duration_minutes = duration_seconds / 60 if duration_seconds else 0
    print(f"Tire #{tire_id}: Duration={duration_seconds}s ({duration_minutes:.1f} min), Status={status}")
    
    # Mark tires with less than 5 minutes (300 seconds) or 0 seconds for deletion
    if duration_seconds is not None and duration_seconds < 300:
        tires_to_delete_nov13.append(tire_id)
        print(f"  --> MARKED FOR DELETION (duration < 5 minutes)")

print(f"\n--- Summary for 2025-11-13 ---")
print(f"Total tires: {len(tires_nov13)}")
print(f"Tires to delete (< 5 min): {len(tires_to_delete_nov13)}")
print(f"Tire IDs to delete: {tires_to_delete_nov13}")

if tires_to_delete_nov13:
    print(f"\nDeleting {len(tires_to_delete_nov13)} tires from 2025-11-13...")
    
    # Delete production logs first (foreign key)
    for tire_id in tires_to_delete_nov13:
        cursor.execute("DELETE FROM production_log WHERE tire_id = ?", (tire_id,))
        print(f"  Deleted production logs for Tire #{tire_id}")
    
    # Delete tires
    cursor.execute(f"DELETE FROM tire_production WHERE id IN ({','.join('?' * len(tires_to_delete_nov13))})", tires_to_delete_nov13)
    
    conn.commit()
    print(f"\nSUCCESS: Deleted {len(tires_to_delete_nov13)} tires from 2025-11-13!")
else:
    print("\nNo tires need deletion on 2025-11-13")

print("\n" + "=" * 80)
print("PART 2: REASSIGN SEQUENTIAL IDs TO ALL TIRES")
print("=" * 80)

# Get all remaining tires in chronological order
cursor.execute("""
    SELECT id, start_time, created_at 
    FROM tire_production 
    ORDER BY created_at, start_time
""")

all_tires = cursor.fetchall()
print(f"\nFound {len(all_tires)} total tires in database")
print("\nCurrent tire IDs (first 10):")
for i, (tire_id, start_time, created_at) in enumerate(all_tires[:10]):
    print(f"  Position {i+1}: Tire #{tire_id} (created: {created_at})")
if len(all_tires) > 10:
    print(f"  ... and {len(all_tires) - 10} more")

print("\n" + "-" * 80)
print("REINDEXING PROCESS")
print("-" * 80)

# Create temporary table with new sequential IDs
cursor.execute("""
    CREATE TEMPORARY TABLE tire_production_temp AS 
    SELECT * FROM tire_production ORDER BY created_at, start_time
""")

# Create ID mapping
cursor.execute("SELECT id FROM tire_production ORDER BY created_at, start_time")
old_ids = [row[0] for row in cursor.fetchall()]

id_mapping = {}
for new_id, old_id in enumerate(old_ids, start=1):
    id_mapping[old_id] = new_id

print(f"\nID Mapping (showing first 15):")
for i, (old_id, new_id) in enumerate(list(id_mapping.items())[:15]):
    print(f"  Old ID {old_id} --> New ID {new_id}")
if len(id_mapping) > 15:
    print(f"  ... and {len(id_mapping) - 15} more mappings")

# Update production_log foreign keys first
print("\nUpdating production_log foreign keys...")
for old_id, new_id in id_mapping.items():
    if old_id != new_id:  # Only update if changed
        # Use a large negative offset to avoid collision during updates
        cursor.execute("UPDATE production_log SET tire_id = ? WHERE tire_id = ?", (-old_id, old_id))

# Now update to final new IDs
for old_id, new_id in id_mapping.items():
    cursor.execute("UPDATE production_log SET tire_id = ? WHERE tire_id = ?", (new_id, -old_id))

print("  Production logs updated!")

# Update tire_production IDs
print("\nUpdating tire_production IDs...")

# First, shift all IDs to negative to avoid collisions
for old_id in old_ids:
    cursor.execute("UPDATE tire_production SET id = ? WHERE id = ?", (-old_id, old_id))

# Then update to new sequential IDs
for old_id, new_id in id_mapping.items():
    cursor.execute("UPDATE tire_production SET id = ? WHERE id = ?", (new_id, -old_id))

print("  Tire IDs updated!")

# Reset SQLite auto-increment counter
max_id = len(all_tires)
# Check if sqlite_sequence table exists (it's created automatically when using AUTOINCREMENT)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
if cursor.fetchone():
    cursor.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'tire_production'", (max_id,))
    print(f"  Auto-increment counter reset to {max_id}")
else:
    print(f"  Note: sqlite_sequence table doesn't exist yet (will be created on next insert)")

conn.commit()

print("\n" + "-" * 80)
print("VERIFICATION")
print("-" * 80)

# Verify the new IDs
cursor.execute("""
    SELECT id, start_time, created_at, duration_seconds 
    FROM tire_production 
    ORDER BY id
    LIMIT 15
""")

print("\nNew sequential IDs (first 15):")
for tire_id, start_time, created_at, duration_seconds in cursor.fetchall():
    print(f"  Tire #{tire_id}: Created {created_at}, Duration {duration_seconds}s")

cursor.execute("SELECT COUNT(*) FROM tire_production")
total = cursor.fetchone()[0]
print(f"\nTotal tires in database: {total}")
print(f"Next auto-increment ID will be: {total + 1}")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"[OK] Cleaned up short-duration tires from 2025-11-13")
print(f"[OK] Reassigned sequential IDs (1 to {total})")
print(f"[OK] Updated all foreign keys in production_log")
print(f"[OK] Reset auto-increment counter")

conn.close()

