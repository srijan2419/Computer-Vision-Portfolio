"""
Utility script to identify and clean up orphaned tire production records
Run this manually to fix phantom tire issues
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import app, db, TireProduction, ProductionLog


def list_orphaned_tires():
    """List all tires stuck in 'in_process' status"""
    with app.app_context():
        orphaned = TireProduction.query.filter_by(status='in_process').all()
        
        if not orphaned:
            print("✓ No orphaned tires found!")
            return []
        
        print(f"Found {len(orphaned)} orphaned tire(s):\n")
        for tire in orphaned:
            duration = (datetime.now() - tire.start_time).total_seconds() / 60
            detection_count = ProductionLog.query.filter_by(tire_id=tire.id).count()
            
            print(f"Tire #{tire.id}:")
            print(f"  Status: {tire.status}")
            print(f"  Started: {tire.start_time}")
            print(f"  Duration: {duration:.1f} minutes (stuck)")
            print(f"  Detections: {detection_count}")
            print(f"  Photos: {tire.photos or 'None'}")
            print()
        
        return orphaned


def complete_tire(tire_id, dry_run=True):
    """
    Manually complete a tire that's stuck in_process
    
    Args:
        tire_id: ID of the tire to complete
        dry_run: If True, only show what would be done without actually doing it
    """
    with app.app_context():
        tire = TireProduction.query.get(tire_id)
        
        if not tire:
            print(f"❌ Tire #{tire_id} not found!")
            return False
        
        if tire.status != 'in_process':
            print(f"❌ Tire #{tire_id} is not in_process (status: {tire.status})")
            return False
        
        # Calculate stats
        logs = ProductionLog.query.filter_by(tire_id=tire.id).all()
        duration = (datetime.now() - tire.start_time).seconds
        
        print(f"Tire #{tire_id}:")
        print(f"  Current status: {tire.status}")
        print(f"  Start time: {tire.start_time}")
        print(f"  Duration: {duration} seconds")
        print(f"  Detection logs: {len(logs)}")
        
        if dry_run:
            print(f"\n[DRY RUN] Would complete tire #{tire_id}")
            print(f"  Would set end_time: {datetime.now()}")
            print(f"  Would set duration_seconds: {duration}")
            print(f"  Would set status: 'completed'")
            if logs:
                avg_conf = sum([log.confidence for log in logs]) / len(logs)
                print(f"  Would set confidence_avg: {avg_conf:.3f}")
            print(f"  Photos: {tire.photos or 'None (would remain None)'}")
            return False
        
        # Actually complete the tire
        tire.end_time = datetime.now()
        tire.duration_seconds = duration
        tire.status = 'completed'
        
        if logs:
            tire.confidence_avg = sum([log.confidence for log in logs]) / len(logs)
        
        db.session.commit()
        
        print(f"\n✓ Successfully completed Tire #{tire_id}")
        return True


def delete_tire(tire_id, dry_run=True):
    """
    Delete a phantom tire and all its associated detection logs
    Use with caution!
    
    Args:
        tire_id: ID of the tire to delete
        dry_run: If True, only show what would be done without actually doing it
    """
    with app.app_context():
        tire = TireProduction.query.get(tire_id)
        
        if not tire:
            print(f"❌ Tire #{tire_id} not found!")
            return False
        
        logs = ProductionLog.query.filter_by(tire_id=tire.id).all()
        
        print(f"Tire #{tire_id}:")
        print(f"  Status: {tire.status}")
        print(f"  Start time: {tire.start_time}")
        print(f"  Detection logs: {len(logs)}")
        print(f"  Photos: {tire.photos or 'None'}")
        
        if dry_run:
            print(f"\n[DRY RUN] Would DELETE tire #{tire_id} and {len(logs)} detection logs")
            print("  ⚠️  This action cannot be undone!")
            return False
        
        # Delete detection logs first (foreign key constraint)
        for log in logs:
            db.session.delete(log)
        
        # Delete the tire
        db.session.delete(tire)
        db.session.commit()
        
        print(f"\n✓ Successfully deleted Tire #{tire_id} and {len(logs)} detection logs")
        return True


def main():
    """Main CLI interface"""
    print("="*60)
    print("Tire Production Orphan Cleanup Utility")
    print("="*60)
    print()
    
    if len(sys.argv) == 1:
        # No arguments - just list orphaned tires
        list_orphaned_tires()
        print("\nUsage:")
        print("  python cleanup_orphans.py                  # List orphaned tires")
        print("  python cleanup_orphans.py complete <id>    # Complete a tire (dry-run)")
        print("  python cleanup_orphans.py complete <id> --confirm  # Actually complete")
        print("  python cleanup_orphans.py delete <id>      # Delete a tire (dry-run)")
        print("  python cleanup_orphans.py delete <id> --confirm    # Actually delete")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_orphaned_tires()
    
    elif command == "complete" and len(sys.argv) >= 3:
        tire_id = int(sys.argv[2])
        dry_run = "--confirm" not in sys.argv
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made\n")
        
        complete_tire(tire_id, dry_run=dry_run)
        
        if dry_run:
            print("\n💡 To actually complete this tire, add --confirm flag")
    
    elif command == "delete" and len(sys.argv) >= 3:
        tire_id = int(sys.argv[2])
        dry_run = "--confirm" not in sys.argv
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made\n")
        
        delete_tire(tire_id, dry_run=dry_run)
        
        if dry_run:
            print("\n💡 To actually delete this tire, add --confirm flag")
            print("⚠️  WARNING: This will permanently delete the tire and all its data!")
    
    else:
        print("Invalid command or missing arguments")
        print("\nUsage:")
        print("  python cleanup_orphans.py list")
        print("  python cleanup_orphans.py complete <id> [--confirm]")
        print("  python cleanup_orphans.py delete <id> [--confirm]")


if __name__ == "__main__":
    main()

